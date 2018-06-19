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

#include "GroupByAndAggregate.h"
#include "AggregateUtils.h"

#include "CardinalityEstimator.h"
#include "ExpressionRange.h"
#include "ExpressionRewrite.h"
#include "GpuInitGroups.h"
#include "InPlaceSort.h"
#include "LLVMFunctionAttributesUtil.h"
#include "MaxwellCodegenPatch.h"
#include "OutputBufferInitialization.h"
#include "ScalarExprVisitor.h"

#include "../CudaMgr/CudaMgr.h"
#include "../Shared/checked_alloc.h"
#include "../Utils/ChunkIter.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "QueryTemplateGenerator.h"
#include "ResultRows.h"
#include "RuntimeFunctions.h"
#include "SpeculativeTopN.h"
#include "StreamingTopN.h"
#include "TopKSort.h"

#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#ifdef ENABLE_COMPACTION
#include <llvm/IR/MDBuilder.h>
#endif

#include <numeric>
#include <thread>

bool g_cluster{false};
bool g_use_result_set{true};
bool g_bigint_count{false};
int g_hll_precision_bits{11};
bool g_enable_smem_group_by{true};
extern size_t g_leaf_count;

namespace {

void check_total_bitmap_memory(const CountDistinctDescriptors& count_distinct_descriptors,
                               const int32_t groups_buffer_entry_count) {
  if (g_enable_watchdog) {
    checked_int64_t total_bytes_per_group = 0;
    for (const auto& count_distinct_desc : count_distinct_descriptors) {
      if (count_distinct_desc.impl_type_ != CountDistinctImplType::Bitmap) {
        continue;
      }
      total_bytes_per_group += count_distinct_desc.bitmapPaddedSizeBytes();
    }
    int64_t total_bytes{0};
    // Need to use OutOfHostMemory since it's the only type of exception
    // QueryExecutionContext is supposed to throw.
    try {
      total_bytes = static_cast<int64_t>(total_bytes_per_group * groups_buffer_entry_count);
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

}  // namespace

int64_t* alloc_group_by_buffer(const size_t numBytes, RenderAllocatorMap* render_allocator_map) {
  if (render_allocator_map) {
    // NOTE(adb): If we got here, we are performing an in-situ rendering query and are not using CUDA buffers.
    // Therefore we need to allocate result set storage using CPU memory.
    const auto gpu_idx = 0;  // Only 1 GPU supported in CUDA-disabled rendering mode
    auto render_allocator_ptr = render_allocator_map->getRenderAllocator(gpu_idx);
    return reinterpret_cast<int64_t*>(render_allocator_ptr->alloc(numBytes));
  } else {
    return reinterpret_cast<int64_t*>(checked_malloc(numBytes));
  }
}

QueryExecutionContext::QueryExecutionContext(const RelAlgExecutionUnit& ra_exe_unit,
                                             const QueryMemoryDescriptor& query_mem_desc,
                                             const std::vector<int64_t>& init_agg_vals,
                                             const Executor* executor,
                                             const ExecutorDeviceType device_type,
                                             const int device_id,
                                             const std::vector<std::vector<const int8_t*>>& col_buffers,
                                             const std::vector<std::vector<const int8_t*>>& iter_buffers,
                                             const std::vector<std::vector<uint64_t>>& frag_offsets,
                                             std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                             const bool output_columnar,
                                             const bool sort_on_gpu,
                                             RenderInfo* render_info)
    : query_mem_desc_(query_mem_desc),
      init_agg_vals_(executor->plan_state_->init_agg_vals_),
      executor_(executor),
      device_type_(device_type),
      device_id_(device_id),
      col_buffers_(col_buffers),
      iter_buffers_(iter_buffers),
      frag_offsets_(frag_offsets),
      consistent_frag_sizes_(get_consistent_frags_sizes(frag_offsets)),
      num_buffers_{device_type == ExecutorDeviceType::CPU
                       ? 1
                       : executor->blockSize() * (query_mem_desc_.blocksShareMemory() ? 1 : executor->gridSize())},
      row_set_mem_owner_(row_set_mem_owner),
      output_columnar_(output_columnar),
      sort_on_gpu_(sort_on_gpu),
      count_distinct_bitmap_mem_(0),
      count_distinct_bitmap_host_mem_(nullptr),
      count_distinct_bitmap_crt_ptr_(nullptr) {
  CHECK(!sort_on_gpu_ || output_columnar);
  if (consistent_frag_sizes_.empty()) {
    // No fragments in the input, no underlying buffers will be needed.
    return;
  }
  check_total_bitmap_memory(query_mem_desc_.count_distinct_descriptors_,
                            query_mem_desc_.entry_count + query_mem_desc_.entry_count_small);
  if (device_type_ == ExecutorDeviceType::GPU) {
    allocateCountDistinctGpuMem();
  }

  auto render_allocator_map =
      render_info && render_info->isPotentialInSituRender() ? render_info->render_allocator_map_ptr.get() : nullptr;
  if (render_allocator_map || query_mem_desc_.group_col_widths.empty()) {
    allocateCountDistinctBuffers(false);
    if (render_info && render_info->useCudaBuffers()) {
      return;
    }
  }

  if (ra_exe_unit.estimator) {
    return;
  }

  const auto thread_count = device_type == ExecutorDeviceType::GPU ? executor->blockSize() * executor->gridSize() : 1;
  const auto group_buffer_size = query_mem_desc_.getBufferSizeBytes(ra_exe_unit, thread_count, device_type);
  OOM_TRACE_PUSH(+": group_buffer_size " + std::to_string(group_buffer_size));
  std::unique_ptr<int64_t, CheckedAllocDeleter> group_by_buffer_template(
      static_cast<int64_t*>(checked_malloc(group_buffer_size)));
  if (!query_mem_desc_.lazyInitGroups(device_type)) {
    if (output_columnar_) {
      initColumnarGroups(
          group_by_buffer_template.get(), &init_agg_vals[0], query_mem_desc_.entry_count, query_mem_desc_.keyless_hash);
    } else {
      auto rows_ptr = group_by_buffer_template.get();
      auto actual_entry_count = query_mem_desc_.entry_count;
      auto warp_size = query_mem_desc_.interleavedBins(device_type_) ? executor_->warpSize() : 1;
      if (use_streaming_top_n(ra_exe_unit, query_mem_desc)) {
        const auto node_count_size = thread_count * sizeof(int64_t);
        memset(rows_ptr, 0, node_count_size);
        const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
        const auto rows_offset = streaming_top_n::get_rows_offset_of_heaps(n, thread_count);
        memset(rows_ptr + thread_count, -1, rows_offset - node_count_size);
        rows_ptr += rows_offset / sizeof(int64_t);
        actual_entry_count = n * thread_count;
        warp_size = 1;
      }
      initGroups(rows_ptr, &init_agg_vals[0], actual_entry_count, query_mem_desc_.keyless_hash, warp_size);
    }
  }

  if (query_mem_desc_.interleavedBins(device_type_)) {
    CHECK(query_mem_desc_.keyless_hash);
  }

  if (query_mem_desc_.keyless_hash) {
    CHECK_EQ(size_t(0), query_mem_desc_.getSmallBufferSizeQuad());
  }

  std::unique_ptr<int64_t, CheckedAllocDeleter> group_by_small_buffer_template;
  if (query_mem_desc_.getSmallBufferSizeBytes()) {
    CHECK(!output_columnar_ && !query_mem_desc_.keyless_hash);
    OOM_TRACE_PUSH(+": getSmallBufferSizeBytes " + std::to_string(query_mem_desc_.getSmallBufferSizeBytes()));
    group_by_small_buffer_template.reset(
        static_cast<int64_t*>(checked_malloc(query_mem_desc_.getSmallBufferSizeBytes())));
    initGroups(group_by_small_buffer_template.get(), &init_agg_vals[0], query_mem_desc_.entry_count_small, false, 1);
  }

  const auto step = device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory() &&
                            !query_mem_desc_.group_col_widths.empty()
                        ? executor_->blockSize()
                        : size_t(1);
  const auto index_buffer_qw = device_type_ == ExecutorDeviceType::GPU && sort_on_gpu_ && query_mem_desc_.keyless_hash
                                   ? query_mem_desc_.entry_count
                                   : size_t(0);
  const auto actual_group_buffer_size = group_buffer_size + index_buffer_qw * sizeof(int64_t);
  const auto actual_small_buffer_size = query_mem_desc_.getSmallBufferSizeBytes();
  const auto group_buffers_count = query_mem_desc_.group_col_widths.empty() ? 1 : num_buffers_;
  for (size_t i = 0; i < group_buffers_count; i += step) {
    OOM_TRACE_PUSH(+": group_by_buffer " + std::to_string(actual_group_buffer_size + actual_small_buffer_size));
    auto group_by_buffer =
        alloc_group_by_buffer(actual_group_buffer_size + actual_small_buffer_size, render_allocator_map);
    if (!query_mem_desc_.lazyInitGroups(device_type)) {
      memcpy(group_by_buffer + index_buffer_qw, group_by_buffer_template.get(), group_buffer_size);
    }
    if (!render_allocator_map) {
      row_set_mem_owner_->addGroupByBuffer(group_by_buffer);
    }
    group_by_buffers_.push_back(group_by_buffer);
    for (size_t j = 1; j < step; ++j) {
      group_by_buffers_.push_back(nullptr);
    }
    if (actual_small_buffer_size) {
      auto group_by_small_buffer = &group_by_buffer[actual_group_buffer_size / sizeof(int64_t)];
      memcpy(group_by_small_buffer, group_by_small_buffer_template.get(), actual_small_buffer_size);
      small_group_by_buffers_.push_back(group_by_small_buffer);
      for (size_t j = 1; j < step; ++j) {
        small_group_by_buffers_.push_back(nullptr);
      }
    }
#ifdef ENABLE_MULTIFRAG_JOIN
    const auto column_frag_offsets = get_col_frag_offsets(ra_exe_unit.target_exprs, frag_offsets);
    const auto column_frag_sizes = get_consistent_frags_sizes(ra_exe_unit.target_exprs, consistent_frag_sizes_);
#endif
    result_sets_.emplace_back(new ResultSet(target_exprs_to_infos(ra_exe_unit.target_exprs, query_mem_desc_),
                                            getColLazyFetchInfo(ra_exe_unit.target_exprs),
                                            col_buffers,
#ifdef ENABLE_MULTIFRAG_JOIN
                                            column_frag_offsets,
                                            column_frag_sizes,
#endif
                                            device_type_,
                                            device_id,
                                            ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_),
                                            row_set_mem_owner_,
                                            executor));
    result_sets_.back()->allocateStorage(reinterpret_cast<int8_t*>(group_by_buffer),
                                         executor_->plan_state_->init_agg_vals_);
    for (size_t j = 1; j < step; ++j) {
      result_sets_.emplace_back(nullptr);
    }
  }
}

namespace {

bool countDescriptorsLogicallyEmpty(const CountDistinctDescriptors& count_distinct_descriptors) {
  return std::all_of(
      count_distinct_descriptors.begin(), count_distinct_descriptors.end(), [](const CountDistinctDescriptor& desc) {
        return desc.impl_type_ == CountDistinctImplType::Invalid;
      });
}

}  // namespace

void QueryExecutionContext::allocateCountDistinctGpuMem() {
  if (countDescriptorsLogicallyEmpty(query_mem_desc_.count_distinct_descriptors_)) {
    return;
  }
  CHECK(executor_);
  auto data_mgr = &executor_->catalog_->get_dataMgr();
  size_t total_bytes_per_entry{0};
  for (const auto& count_distinct_desc : query_mem_desc_.count_distinct_descriptors_) {
    if (count_distinct_desc.impl_type_ == CountDistinctImplType::Invalid) {
      continue;
    }
    CHECK(count_distinct_desc.impl_type_ == CountDistinctImplType::Bitmap);
    total_bytes_per_entry += count_distinct_desc.bitmapPaddedSizeBytes();
  }
  count_distinct_bitmap_mem_bytes_ =
      total_bytes_per_entry * (query_mem_desc_.entry_count + query_mem_desc_.entry_count_small);
  count_distinct_bitmap_mem_ = alloc_gpu_mem(data_mgr, count_distinct_bitmap_mem_bytes_, device_id_, nullptr);
  data_mgr->cudaMgr_->zeroDeviceMem(
      reinterpret_cast<int8_t*>(count_distinct_bitmap_mem_), count_distinct_bitmap_mem_bytes_, device_id_);
  OOM_TRACE_PUSH(+": count_distinct_bitmap_mem_bytes_ " + std::to_string(count_distinct_bitmap_mem_bytes_));
  count_distinct_bitmap_crt_ptr_ = count_distinct_bitmap_host_mem_ =
      static_cast<int8_t*>(checked_malloc(count_distinct_bitmap_mem_bytes_));
  row_set_mem_owner_->addCountDistinctBuffer(count_distinct_bitmap_host_mem_, count_distinct_bitmap_mem_bytes_, true);
}

std::vector<ColumnLazyFetchInfo> QueryExecutionContext::getColLazyFetchInfo(
    const std::vector<Analyzer::Expr*>& target_exprs) const {
  std::vector<ColumnLazyFetchInfo> col_lazy_fetch_info;
  for (const auto target_expr : target_exprs) {
    if (!executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{false, -1, SQLTypeInfo(kNULLT, false)});
    } else {
      const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
      CHECK(col_var);
      auto col_id = col_var->get_column_id();
      auto rte_idx = (col_var->get_rte_idx() == -1) ? 0 : col_var->get_rte_idx();
      auto cd = (col_var->get_table_id() > 0)
                    ? get_column_descriptor(col_id, col_var->get_table_id(), *executor_->catalog_)
                    : nullptr;
      if (cd && IS_GEO(cd->columnType.get_type())) {
        // Geo coords cols will be processed in sequence. So we only need to track the first coords col in lazy fetch
        // info.
        for (auto i = 0; i < cd->columnType.get_physical_coord_cols(); i++) {
          auto cd0 = get_column_descriptor(col_id + i + 1, col_var->get_table_id(), *executor_->catalog_);
          auto col0_ti = cd0->columnType;
          CHECK(!cd0->isVirtualCol);
          auto col0_var = makeExpr<Analyzer::ColumnVar>(col0_ti, col_var->get_table_id(), cd0->columnId, rte_idx);
          auto local_col0_id = executor_->getLocalColumnId(col0_var.get(), false);
          col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col0_id, col0_ti});
          break;
        }
      } else {
        auto local_col_id = executor_->getLocalColumnId(col_var, false);
        const auto& col_ti = col_var->get_type_info();
        col_lazy_fetch_info.emplace_back(ColumnLazyFetchInfo{true, local_col_id, col_ti});
      }
    }
  }
  return col_lazy_fetch_info;
}

void QueryExecutionContext::initColumnPerRow(const QueryMemoryDescriptor& query_mem_desc,
                                             int8_t* row_ptr,
                                             const size_t bin,
                                             const int64_t* init_vals,
                                             const std::vector<ssize_t>& bitmap_sizes) {
  int8_t* col_ptr = row_ptr;
  size_t init_vec_idx = 0;
  for (size_t col_idx = 0; col_idx < query_mem_desc.agg_col_widths.size();
       col_ptr += query_mem_desc.getNextColOffInBytes(col_ptr, bin, col_idx++)) {
    const ssize_t bm_sz{bitmap_sizes[col_idx]};
    int64_t init_val{0};
    if (!bm_sz || query_mem_desc.group_col_widths.empty()) {
      if (query_mem_desc.agg_col_widths[col_idx].compact > 0) {
        init_val = init_vals[init_vec_idx++];
      }
    } else {
      CHECK_EQ(static_cast<size_t>(query_mem_desc.agg_col_widths[col_idx].compact), sizeof(int64_t));
      init_val = bm_sz > 0 ? allocateCountDistinctBitmap(bm_sz) : allocateCountDistinctSet();
      ++init_vec_idx;
    }
    switch (query_mem_desc.agg_col_widths[col_idx].compact) {
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

void QueryExecutionContext::initGroups(int64_t* groups_buffer,
                                       const int64_t* init_vals,
                                       const int32_t groups_buffer_entry_count,
                                       const bool keyless,
                                       const size_t warp_size) {
  const size_t key_count{query_mem_desc_.group_col_widths.size()};
  const size_t row_size{query_mem_desc_.getRowSize()};
  const size_t col_base_off{query_mem_desc_.getColOffInBytes(0, 0)};

  auto agg_bitmap_size = allocateCountDistinctBuffers(true);
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);

  const auto query_mem_desc_fixedup = ResultSet::fixupQueryMemoryDescriptor(query_mem_desc_);

  if (keyless) {
    CHECK(warp_size >= 1);
    CHECK(key_count == 1);
    for (size_t warp_idx = 0; warp_idx < warp_size; ++warp_idx) {
      for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count); ++bin, buffer_ptr += row_size) {
        initColumnPerRow(query_mem_desc_fixedup, &buffer_ptr[col_base_off], bin, init_vals, agg_bitmap_size);
      }
    }
    return;
  }

  for (size_t bin = 0; bin < static_cast<size_t>(groups_buffer_entry_count); ++bin, buffer_ptr += row_size) {
    fill_empty_key(buffer_ptr, key_count, query_mem_desc_.getEffectiveKeyWidth());
    initColumnPerRow(query_mem_desc_fixedup, &buffer_ptr[col_base_off], bin, init_vals, agg_bitmap_size);
  }
}

template <typename T>
int8_t* QueryExecutionContext::initColumnarBuffer(T* buffer_ptr, const T init_val, const uint32_t entry_count) {
  static_assert(sizeof(T) <= sizeof(int64_t), "Unsupported template type");
  for (uint32_t i = 0; i < entry_count; ++i) {
    buffer_ptr[i] = init_val;
  }
  return reinterpret_cast<int8_t*>(buffer_ptr + entry_count);
}

void QueryExecutionContext::initColumnarGroups(int64_t* groups_buffer,
                                               const int64_t* init_vals,
                                               const int32_t groups_buffer_entry_count,
                                               const bool keyless) {
  for (const auto target_expr : executor_->plan_state_->target_exprs_) {
    const auto agg_info = target_info(target_expr);
    CHECK(!is_distinct_target(agg_info));
  }
  const bool need_padding = !query_mem_desc_.isCompactLayoutIsometric();
  const int32_t agg_col_count = query_mem_desc_.agg_col_widths.size();
  const int32_t key_qw_count = query_mem_desc_.group_col_widths.size();
  auto buffer_ptr = reinterpret_cast<int8_t*>(groups_buffer);
  CHECK(key_qw_count == 1);
  if (!keyless) {
    buffer_ptr =
        initColumnarBuffer<int64_t>(reinterpret_cast<int64_t*>(buffer_ptr), EMPTY_KEY_64, groups_buffer_entry_count);
  }
  for (int32_t i = 0; i < agg_col_count; ++i) {
    switch (query_mem_desc_.agg_col_widths[i].compact) {
      case 1:
        buffer_ptr = initColumnarBuffer<int8_t>(buffer_ptr, init_vals[i], groups_buffer_entry_count);
        break;
      case 2:
        buffer_ptr = initColumnarBuffer<int16_t>(
            reinterpret_cast<int16_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count);
        break;
      case 4:
        buffer_ptr = initColumnarBuffer<int32_t>(
            reinterpret_cast<int32_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count);
        break;
      case 8:
        buffer_ptr = initColumnarBuffer<int64_t>(
            reinterpret_cast<int64_t*>(buffer_ptr), init_vals[i], groups_buffer_entry_count);
        break;
      default:
        CHECK(false);
    }
    if (need_padding) {
      buffer_ptr = align_to_int64(buffer_ptr);
    }
  }
}

// deferred is true for group by queries; initGroups will allocate a bitmap
// for each group slot
std::vector<ssize_t> QueryExecutionContext::allocateCountDistinctBuffers(const bool deferred) {
  const size_t agg_col_count{query_mem_desc_.agg_col_widths.size()};
  std::vector<ssize_t> agg_bitmap_size(deferred ? agg_col_count : 0);

  CHECK_GE(agg_col_count, executor_->plan_state_->target_exprs_.size());
  for (size_t target_idx = 0, agg_col_idx = 0;
       target_idx < executor_->plan_state_->target_exprs_.size() && agg_col_idx < agg_col_count;
       ++target_idx, ++agg_col_idx) {
    const auto target_expr = executor_->plan_state_->target_exprs_[target_idx];
    const auto agg_info = target_info(target_expr);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg && (agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT));
      CHECK_EQ(static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_col_idx].actual), sizeof(int64_t));
      CHECK_LT(target_idx, query_mem_desc_.count_distinct_descriptors_.size());
      const auto& count_distinct_desc = query_mem_desc_.count_distinct_descriptors_[target_idx];
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

int64_t QueryExecutionContext::allocateCountDistinctBitmap(const size_t bitmap_byte_sz) {
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

int64_t QueryExecutionContext::allocateCountDistinctSet() {
  auto count_distinct_set = new std::set<int64_t>();
  row_set_mem_owner_->addCountDistinctSet(count_distinct_set);
  return reinterpret_cast<int64_t>(count_distinct_set);
}

RowSetPtr QueryExecutionContext::getRowSet(const RelAlgExecutionUnit& ra_exe_unit,
                                           const QueryMemoryDescriptor& query_mem_desc,
                                           const bool was_auto_device) const {
  std::vector<std::pair<ResultPtr, std::vector<size_t>>> results_per_sm;
  CHECK_EQ(num_buffers_, group_by_buffers_.size());
  if (device_type_ == ExecutorDeviceType::CPU) {
    CHECK_EQ(size_t(1), num_buffers_);
    return groupBufferToResults(0, ra_exe_unit.target_exprs, was_auto_device);
  }
  size_t step{query_mem_desc_.threadsShareMemory() ? executor_->blockSize() : 1};
  for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
    results_per_sm.emplace_back(groupBufferToResults(i, ra_exe_unit.target_exprs, was_auto_device),
                                std::vector<size_t>{});
  }
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return executor_->reduceMultiDeviceResults(ra_exe_unit, results_per_sm, row_set_mem_owner_, query_mem_desc);
}

bool QueryExecutionContext::isEmptyBin(const int64_t* group_by_buffer, const size_t bin, const size_t key_idx) const {
  auto key_ptr = reinterpret_cast<const int8_t*>(group_by_buffer) + query_mem_desc_.getKeyOffInBytes(bin, key_idx);
  switch (query_mem_desc_.getEffectiveKeyWidth()) {
    case 4:
      if (*reinterpret_cast<const int32_t*>(key_ptr) == EMPTY_KEY_32) {
        return true;
      }
      break;
    case 8:
      if (*reinterpret_cast<const int64_t*>(key_ptr) == EMPTY_KEY_64) {
        return true;
      }
      break;
    default:
      CHECK(false);
  }
  return false;
}

#ifdef HAVE_CUDA
void QueryExecutionContext::initializeDynamicWatchdog(void* native_module, const int device_id) const {
  auto cu_module = static_cast<CUmodule>(native_module);
  CHECK(cu_module);
  CUevent start, stop;
  cuEventCreate(&start, 0);
  cuEventCreate(&stop, 0);
  cuEventRecord(start, 0);

  CUdeviceptr dw_cycle_budget;
  size_t dw_cycle_budget_size;
  // Translate milliseconds to device cycles
  uint64_t cycle_budget = executor_->deviceCycles(g_dynamic_watchdog_time_limit);
  if (device_id == 0) {
    LOG(INFO) << "Dynamic Watchdog budget: GPU: " << std::to_string(g_dynamic_watchdog_time_limit) << "ms, "
              << std::to_string(cycle_budget) << " cycles";
  }
  checkCudaErrors(cuModuleGetGlobal(&dw_cycle_budget, &dw_cycle_budget_size, cu_module, "dw_cycle_budget"));
  CHECK_EQ(dw_cycle_budget_size, sizeof(uint64_t));
  checkCudaErrors(cuMemcpyHtoD(dw_cycle_budget, reinterpret_cast<void*>(&cycle_budget), sizeof(uint64_t)));

  CUdeviceptr dw_sm_cycle_start;
  size_t dw_sm_cycle_start_size;
  checkCudaErrors(cuModuleGetGlobal(&dw_sm_cycle_start, &dw_sm_cycle_start_size, cu_module, "dw_sm_cycle_start"));
  CHECK_EQ(dw_sm_cycle_start_size, 64 * sizeof(uint64_t));
  checkCudaErrors(cuMemsetD32(dw_sm_cycle_start, 0, 64 * 2));

  if (!executor_->interrupted_) {
    // Executor is not marked as interrupted, make sure dynamic watchdog doesn't block execution
    CUdeviceptr dw_abort;
    size_t dw_abort_size;
    checkCudaErrors(cuModuleGetGlobal(&dw_abort, &dw_abort_size, cu_module, "dw_abort"));
    CHECK_EQ(dw_abort_size, sizeof(uint32_t));
    checkCudaErrors(cuMemsetD32(dw_abort, 0, 1));
  }

  cuEventRecord(stop, 0);
  cuEventSynchronize(stop);
  float milliseconds = 0;
  cuEventElapsedTime(&milliseconds, start, stop);
  VLOG(1) << "Device " << std::to_string(device_id)
          << ": launchGpuCode: dynamic watchdog init: " << std::to_string(milliseconds) << " ms\n";
}

std::vector<CUdeviceptr> QueryExecutionContext::prepareKernelParams(
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<int8_t>& literal_buff,
    const std::vector<std::vector<int64_t>>& num_rows,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    const uint32_t frag_stride,
    const int32_t scan_limit,
    const std::vector<int64_t>& init_agg_vals,
    const std::vector<int32_t>& error_codes,
    const uint32_t num_tables,
    const std::vector<int64_t>& join_hash_tables,
    Data_Namespace::DataMgr* data_mgr,
    const int device_id,
    const bool hoist_literals,
    const bool is_group_by) const {
  std::vector<CUdeviceptr> params(KERN_PARAM_COUNT, 0);
  const uint32_t num_fragments = col_buffers.size();
  const size_t col_count{num_fragments > 0 ? col_buffers.front().size() : 0};
  if (col_count) {
    std::vector<CUdeviceptr> multifrag_col_dev_buffers;
    for (auto frag_col_buffers : col_buffers) {
      std::vector<CUdeviceptr> col_dev_buffers;
      for (auto col_buffer : frag_col_buffers) {
        col_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(col_buffer));
      }
      auto col_buffers_dev_ptr = alloc_gpu_mem(data_mgr, col_count * sizeof(CUdeviceptr), device_id, nullptr);
      copy_to_gpu(data_mgr, col_buffers_dev_ptr, &col_dev_buffers[0], col_count * sizeof(CUdeviceptr), device_id);
      multifrag_col_dev_buffers.push_back(col_buffers_dev_ptr);
    }
    params[COL_BUFFERS] = alloc_gpu_mem(data_mgr, num_fragments * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(
        data_mgr, params[COL_BUFFERS], &multifrag_col_dev_buffers[0], num_fragments * sizeof(CUdeviceptr), device_id);
  }
  params[NUM_FRAGMENTS] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_FRAGMENTS], &num_fragments, sizeof(uint32_t), device_id);

  params[FRAG_STRIDE] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[FRAG_STRIDE], &frag_stride, sizeof(uint32_t), device_id);

  CUdeviceptr literals_and_addr_mapping =
      alloc_gpu_mem(data_mgr, literal_buff.size() + 2 * sizeof(int64_t), device_id, nullptr);
  CHECK_EQ(0, literals_and_addr_mapping % 8);
  std::vector<int64_t> additional_literal_bytes;
  if (count_distinct_bitmap_mem_) {
    // Store host and device addresses
    CHECK(count_distinct_bitmap_host_mem_);
    additional_literal_bytes.push_back(reinterpret_cast<int64_t>(count_distinct_bitmap_host_mem_));
    additional_literal_bytes.push_back(static_cast<int64_t>(count_distinct_bitmap_mem_));
    copy_to_gpu(data_mgr,
                literals_and_addr_mapping,
                &additional_literal_bytes[0],
                additional_literal_bytes.size() * sizeof(additional_literal_bytes[0]),
                device_id);
  }
  params[LITERALS] = literals_and_addr_mapping + additional_literal_bytes.size() * sizeof(additional_literal_bytes[0]);
  if (!literal_buff.empty()) {
    CHECK(hoist_literals);
    copy_to_gpu(data_mgr, params[LITERALS], &literal_buff[0], literal_buff.size(), device_id);
  }
  CHECK_EQ(num_rows.size(), col_buffers.size());
  std::vector<int64_t> flatened_num_rows;
  for (auto& nums : num_rows) {
    CHECK_EQ(nums.size(), num_tables);
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  params[NUM_ROWS] = alloc_gpu_mem(data_mgr, sizeof(int64_t) * flatened_num_rows.size(), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_ROWS], &flatened_num_rows[0], sizeof(int64_t) * flatened_num_rows.size(), device_id);

  CHECK_EQ(frag_offsets.size(), col_buffers.size());
  std::vector<int64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    CHECK_EQ(offsets.size(), num_tables);
    flatened_frag_offsets.insert(flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  params[FRAG_ROW_OFFSETS] =
      alloc_gpu_mem(data_mgr, sizeof(int64_t) * flatened_frag_offsets.size(), device_id, nullptr);
  copy_to_gpu(data_mgr,
              params[FRAG_ROW_OFFSETS],
              &flatened_frag_offsets[0],
              sizeof(int64_t) * flatened_frag_offsets.size(),
              device_id);
  int32_t max_matched{scan_limit};
  params[MAX_MATCHED] = alloc_gpu_mem(data_mgr, sizeof(max_matched), device_id, nullptr);
  copy_to_gpu(data_mgr, params[MAX_MATCHED], &max_matched, sizeof(max_matched), device_id);

  int32_t total_matched{0};
  params[TOTAL_MATCHED] = alloc_gpu_mem(data_mgr, sizeof(total_matched), device_id, nullptr);
  copy_to_gpu(data_mgr, params[TOTAL_MATCHED], &total_matched, sizeof(total_matched), device_id);

  if (is_group_by && !output_columnar_) {
    auto cmpt_sz = align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t);
    auto cmpt_val_buff = compact_init_vals(cmpt_sz, init_agg_vals, query_mem_desc_.agg_col_widths);
    params[INIT_AGG_VALS] = alloc_gpu_mem(data_mgr, cmpt_sz * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr, params[INIT_AGG_VALS], &cmpt_val_buff[0], cmpt_sz * sizeof(int64_t), device_id);
  } else {
    params[INIT_AGG_VALS] = alloc_gpu_mem(data_mgr, init_agg_vals.size() * sizeof(int64_t), device_id, nullptr);
    copy_to_gpu(data_mgr, params[INIT_AGG_VALS], &init_agg_vals[0], init_agg_vals.size() * sizeof(int64_t), device_id);
  }

  params[ERROR_CODE] = alloc_gpu_mem(data_mgr, error_codes.size() * sizeof(error_codes[0]), device_id, nullptr);
  copy_to_gpu(data_mgr, params[ERROR_CODE], &error_codes[0], error_codes.size() * sizeof(error_codes[0]), device_id);

  params[NUM_TABLES] = alloc_gpu_mem(data_mgr, sizeof(uint32_t), device_id, nullptr);
  copy_to_gpu(data_mgr, params[NUM_TABLES], &num_tables, sizeof(uint32_t), device_id);

  const auto hash_table_count = join_hash_tables.size();
  switch (hash_table_count) {
    case 0: {
      params[JOIN_HASH_TABLES] = CUdeviceptr(0);
    } break;
    case 1:
      params[JOIN_HASH_TABLES] = static_cast<CUdeviceptr>(join_hash_tables[0]);
      break;
    default: {
      params[JOIN_HASH_TABLES] = alloc_gpu_mem(data_mgr, hash_table_count * sizeof(int64_t), device_id, nullptr);
      copy_to_gpu(
          data_mgr, params[JOIN_HASH_TABLES], &join_hash_tables[0], hash_table_count * sizeof(int64_t), device_id);
    } break;
  }

  return params;
}

std::pair<CUdeviceptr, CUdeviceptr> QueryExecutionContext::prepareTopNHeapsDevBuffer(
    Data_Namespace::DataMgr* data_mgr,
    const CUdeviceptr init_agg_vals_dev_ptr,
    const size_t n,
    const int device_id,
    const unsigned block_size_x,
    const unsigned grid_size_x) const {
  const auto thread_count = block_size_x * grid_size_x;
  const auto total_buff_size = streaming_top_n::get_heap_size(query_mem_desc_.getRowSize(), n, thread_count);
  OOM_TRACE_PUSH();
  CUdeviceptr dev_buffer = alloc_gpu_mem(data_mgr, total_buff_size, device_id, nullptr);

  std::vector<CUdeviceptr> dev_buffers(thread_count);

  for (size_t i = 0; i < thread_count; ++i) {
    dev_buffers[i] = dev_buffer;
  }

  auto dev_ptr = alloc_gpu_mem(data_mgr, thread_count * sizeof(CUdeviceptr), device_id, nullptr);
  copy_to_gpu(data_mgr, dev_ptr, &dev_buffers[0], thread_count * sizeof(CUdeviceptr), device_id);

  CHECK(query_mem_desc_.lazyInitGroups(ExecutorDeviceType::GPU));
  CHECK(!output_columnar_);

  data_mgr->cudaMgr_->zeroDeviceMem(reinterpret_cast<int8_t*>(dev_buffer), thread_count * sizeof(int64_t), device_id);
  data_mgr->cudaMgr_->setDeviceMem(reinterpret_cast<int8_t*>(dev_buffer + thread_count * sizeof(int64_t)),
                                   (unsigned char)-1,
                                   thread_count * n * sizeof(int64_t),
                                   device_id);
  init_group_by_buffer_on_device(
      reinterpret_cast<int64_t*>(dev_buffer + streaming_top_n::get_rows_offset_of_heaps(n, thread_count)),
      reinterpret_cast<int64_t*>(init_agg_vals_dev_ptr),
      n * thread_count,
      query_mem_desc_.group_col_widths.size(),
      query_mem_desc_.getEffectiveKeyWidth(),
      query_mem_desc_.getRowSize() / sizeof(int64_t),
      query_mem_desc_.keyless_hash,
      1,
      block_size_x,
      grid_size_x);

  return {dev_ptr, dev_buffer};
}

GpuQueryMemory QueryExecutionContext::prepareGroupByDevBuffer(Data_Namespace::DataMgr* data_mgr,
                                                              RenderAllocator* render_allocator,
                                                              const RelAlgExecutionUnit& ra_exe_unit,
                                                              const CUdeviceptr init_agg_vals_dev_ptr,
                                                              const int device_id,
                                                              const unsigned block_size_x,
                                                              const unsigned grid_size_x,
                                                              const bool can_sort_on_gpu) const {
  if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
    CHECK(!render_allocator);
    const auto n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    auto heap_buffers =
        prepareTopNHeapsDevBuffer(data_mgr, init_agg_vals_dev_ptr, n, device_id, block_size_x, grid_size_x);
    return GpuQueryMemory{heap_buffers};
  }
  auto gpu_query_mem = create_dev_group_by_buffers(data_mgr,
                                                   group_by_buffers_,
                                                   small_group_by_buffers_,
                                                   query_mem_desc_,
                                                   block_size_x,
                                                   grid_size_x,
                                                   device_id,
                                                   can_sort_on_gpu,
                                                   false,
                                                   render_allocator);
  if (render_allocator) {
    CHECK_EQ(size_t(0), render_allocator->getAllocatedSize() % 8);
  }
  if (query_mem_desc_.lazyInitGroups(ExecutorDeviceType::GPU)) {
    CHECK(!render_allocator);
    const size_t step{query_mem_desc_.threadsShareMemory() ? block_size_x : 1};
    size_t groups_buffer_size{query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU)};
    auto group_by_dev_buffer = gpu_query_mem.group_by_buffers.second;
    const size_t col_count = query_mem_desc_.agg_col_widths.size();
    CUdeviceptr col_widths_dev_ptr{0};
    if (output_columnar_) {
      std::vector<int8_t> compact_col_widths(col_count);
      for (size_t idx = 0; idx < col_count; ++idx) {
        compact_col_widths[idx] = query_mem_desc_.agg_col_widths[idx].compact;
      }
      col_widths_dev_ptr = alloc_gpu_mem(data_mgr, col_count * sizeof(int8_t), device_id, nullptr);
      copy_to_gpu(data_mgr, col_widths_dev_ptr, &compact_col_widths[0], col_count * sizeof(int8_t), device_id);
    }
    const int8_t warp_count = query_mem_desc_.interleavedBins(ExecutorDeviceType::GPU) ? executor_->warpSize() : 1;
    OOM_TRACE_PUSH();
    for (size_t i = 0; i < group_by_buffers_.size(); i += step) {
      if (output_columnar_) {
        init_columnar_group_by_buffer_on_device(reinterpret_cast<int64_t*>(group_by_dev_buffer),
                                                reinterpret_cast<const int64_t*>(init_agg_vals_dev_ptr),
                                                query_mem_desc_.entry_count,
                                                query_mem_desc_.group_col_widths.size(),
                                                col_count,
                                                reinterpret_cast<int8_t*>(col_widths_dev_ptr),
                                                !query_mem_desc_.isCompactLayoutIsometric(),
                                                query_mem_desc_.keyless_hash,
                                                sizeof(int64_t),
                                                block_size_x,
                                                grid_size_x);
      } else {
        init_group_by_buffer_on_device(reinterpret_cast<int64_t*>(group_by_dev_buffer),
                                       reinterpret_cast<int64_t*>(init_agg_vals_dev_ptr),
                                       query_mem_desc_.entry_count,
                                       query_mem_desc_.group_col_widths.size(),
                                       query_mem_desc_.getEffectiveKeyWidth(),
                                       query_mem_desc_.getRowSize() / sizeof(int64_t),
                                       query_mem_desc_.keyless_hash,
                                       warp_count,
                                       block_size_x,
                                       grid_size_x);
      }
      group_by_dev_buffer += groups_buffer_size;
    }
  }
  return gpu_query_mem;
}

namespace {

int32_t aggregate_error_codes(const std::vector<int32_t>& error_codes) {
  // Check overflow / division by zero / interrupt first
  for (const auto err : error_codes) {
    if (err > 0) {
      return err;
    }
  }
  for (const auto err : error_codes) {
    if (err) {
      return err;
    }
  }
  return 0;
}

}  // namespace
#endif

std::vector<int64_t*> QueryExecutionContext::launchGpuCode(const RelAlgExecutionUnit& ra_exe_unit,
                                                           const std::vector<std::pair<void*, void*>>& cu_functions,
                                                           const bool hoist_literals,
                                                           const std::vector<int8_t>& literal_buff,
                                                           std::vector<std::vector<const int8_t*>> col_buffers,
                                                           const std::vector<std::vector<int64_t>>& num_rows,
                                                           const std::vector<std::vector<uint64_t>>& frag_offsets,
                                                           const uint32_t frag_stride,
                                                           const int32_t scan_limit,
                                                           const std::vector<int64_t>& init_agg_vals,
                                                           Data_Namespace::DataMgr* data_mgr,
                                                           const unsigned block_size_x,
                                                           const unsigned grid_size_x,
                                                           const int device_id,
                                                           int32_t* error_code,
                                                           const uint32_t num_tables,
                                                           const std::vector<int64_t>& join_hash_tables,
                                                           RenderAllocatorMap* render_allocator_map) {
  INJECT_TIMER(lauchGpuCode);
#ifdef HAVE_CUDA
  bool is_group_by{!query_mem_desc_.group_col_widths.empty()};
  data_mgr->cudaMgr_->setContext(device_id);

  RenderAllocator* render_allocator = nullptr;
  if (render_allocator_map) {
    render_allocator = render_allocator_map->getRenderAllocator(device_id);
  }

  auto cu_func = static_cast<CUfunction>(cu_functions[device_id].first);
  std::vector<int64_t*> out_vec;
  uint32_t num_fragments = col_buffers.size();
  std::vector<int32_t> error_codes(grid_size_x * block_size_x);

  CUevent start0, stop0;  // preparation
  cuEventCreate(&start0, 0);
  cuEventCreate(&stop0, 0);
  CUevent start1, stop1;  // cuLaunchKernel
  cuEventCreate(&start1, 0);
  cuEventCreate(&stop1, 0);
  CUevent start2, stop2;  // finish
  cuEventCreate(&start2, 0);
  cuEventCreate(&stop2, 0);

  if (g_enable_dynamic_watchdog) {
    cuEventRecord(start0, 0);
  }

  if (g_enable_dynamic_watchdog) {
    initializeDynamicWatchdog(cu_functions[device_id].second, device_id);
  }

  auto kernel_params = prepareKernelParams(col_buffers,
                                           literal_buff,
                                           num_rows,
                                           frag_offsets,
                                           frag_stride,
                                           scan_limit,
                                           init_agg_vals,
                                           error_codes,
                                           num_tables,
                                           join_hash_tables,
                                           data_mgr,
                                           device_id,
                                           hoist_literals,
                                           is_group_by);

  CHECK_EQ(static_cast<size_t>(KERN_PARAM_COUNT), kernel_params.size());
  CHECK_EQ(CUdeviceptr(0), kernel_params[GROUPBY_BUF]);
  CHECK_EQ(CUdeviceptr(0), kernel_params[SMALL_BUF]);

  const unsigned block_size_y = 1;
  const unsigned block_size_z = 1;
  const unsigned grid_size_y = 1;
  const unsigned grid_size_z = 1;
  const auto total_thread_count = block_size_x * grid_size_x;
  const auto err_desc = kernel_params[ERROR_CODE];

  if (is_group_by) {
    CHECK(!group_by_buffers_.empty() || render_allocator);
    bool can_sort_on_gpu = query_mem_desc_.sortOnGpu();
    auto gpu_query_mem = prepareGroupByDevBuffer(data_mgr,
                                                 render_allocator,
                                                 ra_exe_unit,
                                                 kernel_params[INIT_AGG_VALS],
                                                 device_id,
                                                 block_size_x,
                                                 grid_size_x,
                                                 can_sort_on_gpu);

    kernel_params[GROUPBY_BUF] = gpu_query_mem.group_by_buffers.first;
    kernel_params[SMALL_BUF] = gpu_query_mem.small_group_by_buffers.first;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog) {
      cuEventRecord(stop0, 0);
      cuEventSynchronize(stop0);
      float milliseconds0 = 0;
      cuEventElapsedTime(&milliseconds0, start0, stop0);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by prepare: " << std::to_string(milliseconds0) << " ms";
      cuEventRecord(start1, 0);
    }

    if (hoist_literals) {
      OOM_TRACE_PUSH();
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    } else {
      OOM_TRACE_PUSH();
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     query_mem_desc_.sharedMemBytes(ExecutorDeviceType::GPU),
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    }
    if (g_enable_dynamic_watchdog) {
      executor_->registerActiveModule(cu_functions[device_id].second, device_id);
      cuEventRecord(stop1, 0);
      cuEventSynchronize(stop1);
      executor_->unregisterActiveModule(cu_functions[device_id].second, device_id);
      float milliseconds1 = 0;
      cuEventElapsedTime(&milliseconds1, start1, stop1);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: group-by cuLaunchKernel: " << std::to_string(milliseconds1) << " ms";
      cuEventRecord(start2, 0);
    }

    copy_from_gpu(data_mgr, &error_codes[0], err_desc, error_codes.size() * sizeof(error_codes[0]), device_id);
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }

    if (!render_allocator) {
      if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
        CHECK_EQ(group_by_buffers_.size(), num_buffers_);
        const auto rows_copy =
            pick_top_n_rows_from_dev_heaps(data_mgr,
                                           reinterpret_cast<int64_t*>(gpu_query_mem.group_by_buffers.second),
                                           ra_exe_unit,
                                           query_mem_desc_,
                                           total_thread_count,
                                           device_id);
        CHECK_EQ(rows_copy.size(), static_cast<size_t>(query_mem_desc_.entry_count * query_mem_desc_.getRowSize()));
        memcpy(group_by_buffers_[0], &rows_copy[0], rows_copy.size());
      } else {
        if (use_speculative_top_n(ra_exe_unit, query_mem_desc_)) {
          ResultRows::inplaceSortGpuImpl(
              ra_exe_unit.sort_info.order_entries, query_mem_desc_, gpu_query_mem, data_mgr, device_id);
        }
        copy_group_by_buffers_from_gpu(data_mgr,
                                       this,
                                       gpu_query_mem,
                                       ra_exe_unit,
                                       block_size_x,
                                       grid_size_x,
                                       device_id,
                                       can_sort_on_gpu && query_mem_desc_.keyless_hash);
      }
    }
  } else {
    CHECK_EQ(num_fragments % frag_stride, 0u);
    const auto num_out_frags = num_fragments / frag_stride;
    std::vector<CUdeviceptr> out_vec_dev_buffers;
    const size_t agg_col_count{ra_exe_unit.estimator ? size_t(1) : init_agg_vals.size()};
    if (ra_exe_unit.estimator) {
      estimator_result_set_.reset(new ResultSet(ra_exe_unit.estimator, ExecutorDeviceType::GPU, device_id, data_mgr));
      out_vec_dev_buffers.push_back(reinterpret_cast<CUdeviceptr>(estimator_result_set_->getDeviceEstimatorBuffer()));
    } else {
      OOM_TRACE_PUSH();
      for (size_t i = 0; i < agg_col_count; ++i) {
        auto out_vec_dev_buffer =
            num_out_frags
                ? alloc_gpu_mem(
                      data_mgr, block_size_x * grid_size_x * sizeof(int64_t) * num_out_frags, device_id, nullptr)
                : 0;
        out_vec_dev_buffers.push_back(out_vec_dev_buffer);
      }
    }
    auto out_vec_dev_ptr = alloc_gpu_mem(data_mgr, agg_col_count * sizeof(CUdeviceptr), device_id, nullptr);
    copy_to_gpu(data_mgr, out_vec_dev_ptr, &out_vec_dev_buffers[0], agg_col_count * sizeof(CUdeviceptr), device_id);
    CUdeviceptr unused_dev_ptr{0};
    kernel_params[GROUPBY_BUF] = out_vec_dev_ptr;
    kernel_params[SMALL_BUF] = unused_dev_ptr;
    std::vector<void*> param_ptrs;
    for (auto& param : kernel_params) {
      param_ptrs.push_back(&param);
    }

    if (g_enable_dynamic_watchdog) {
      cuEventRecord(stop0, 0);
      cuEventSynchronize(stop0);
      float milliseconds0 = 0;
      cuEventElapsedTime(&milliseconds0, start0, stop0);
      VLOG(1) << "Device " << std::to_string(device_id) << ": launchGpuCode: prepare: " << std::to_string(milliseconds0)
              << " ms";
      cuEventRecord(start1, 0);
    }

    if (hoist_literals) {
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    } else {
      param_ptrs.erase(param_ptrs.begin() + LITERALS);  // TODO(alex): remove
      checkCudaErrors(cuLaunchKernel(cu_func,
                                     grid_size_x,
                                     grid_size_y,
                                     grid_size_z,
                                     block_size_x,
                                     block_size_y,
                                     block_size_z,
                                     0,
                                     nullptr,
                                     &param_ptrs[0],
                                     nullptr));
    }

    if (g_enable_dynamic_watchdog) {
      executor_->registerActiveModule(cu_functions[device_id].second, device_id);
      cuEventRecord(stop1, 0);
      cuEventSynchronize(stop1);
      executor_->unregisterActiveModule(cu_functions[device_id].second, device_id);
      float milliseconds1 = 0;
      cuEventElapsedTime(&milliseconds1, start1, stop1);
      VLOG(1) << "Device " << std::to_string(device_id)
              << ": launchGpuCode: cuLaunchKernel: " << std::to_string(milliseconds1) << " ms";
      cuEventRecord(start2, 0);
    }

    copy_from_gpu(data_mgr, &error_codes[0], err_desc, error_codes.size() * sizeof(error_codes[0]), device_id);
    *error_code = aggregate_error_codes(error_codes);
    if (*error_code > 0) {
      return {};
    }
    if (ra_exe_unit.estimator) {
      CHECK(estimator_result_set_);
      estimator_result_set_->syncEstimatorBuffer();
      return {};
    }
    for (size_t i = 0; i < agg_col_count; ++i) {
      int64_t* host_out_vec = new int64_t[block_size_x * grid_size_x * sizeof(int64_t) * num_out_frags];
      copy_from_gpu(data_mgr,
                    host_out_vec,
                    out_vec_dev_buffers[i],
                    block_size_x * grid_size_x * sizeof(int64_t) * num_out_frags,
                    device_id);
      out_vec.push_back(host_out_vec);
    }
  }
  if (count_distinct_bitmap_mem_) {
    copy_from_gpu(data_mgr,
                  count_distinct_bitmap_host_mem_,
                  count_distinct_bitmap_mem_,
                  count_distinct_bitmap_mem_bytes_,
                  device_id);
  }

  if (g_enable_dynamic_watchdog) {
    cuEventRecord(stop2, 0);
    cuEventSynchronize(stop2);
    float milliseconds2 = 0;
    cuEventElapsedTime(&milliseconds2, start2, stop2);
    VLOG(1) << "Device " << std::to_string(device_id) << ": launchGpuCode: finish: " << std::to_string(milliseconds2)
            << " ms";
  }

  return out_vec;
#else
  return {};
#endif
}

std::vector<int64_t*> QueryExecutionContext::launchCpuCode(const RelAlgExecutionUnit& ra_exe_unit,
                                                           const std::vector<std::pair<void*, void*>>& fn_ptrs,
                                                           const bool hoist_literals,
                                                           const std::vector<int8_t>& literal_buff,
                                                           std::vector<std::vector<const int8_t*>> col_buffers,
                                                           const std::vector<std::vector<int64_t>>& num_rows,
                                                           const std::vector<std::vector<uint64_t>>& frag_offsets,
                                                           const uint32_t frag_stride,
                                                           const int32_t scan_limit,
                                                           const std::vector<int64_t>& init_agg_vals,
                                                           int32_t* error_code,
                                                           const uint32_t num_tables,
                                                           const std::vector<int64_t>& join_hash_tables) {
  INJECT_TIMER(lauchCpuCode);
  std::vector<const int8_t**> multifrag_col_buffers;
  for (auto& col_buffer : col_buffers) {
    multifrag_col_buffers.push_back(&col_buffer[0]);
  }
  const int8_t*** multifrag_cols_ptr{multifrag_col_buffers.empty() ? nullptr : &multifrag_col_buffers[0]};
  int64_t** small_group_by_buffers_ptr{small_group_by_buffers_.empty() ? nullptr : &small_group_by_buffers_[0]};
  const uint32_t num_fragments =
      multifrag_cols_ptr ? static_cast<uint32_t>(col_buffers.size()) : 0u;  // TODO(miyu): check 0
  CHECK_EQ(num_fragments % frag_stride, 0u);
  const auto num_out_frags = multifrag_cols_ptr ? num_fragments / frag_stride : 0u;

  const bool is_group_by{!query_mem_desc_.group_col_widths.empty()};
  std::vector<int64_t*> out_vec;
  if (ra_exe_unit.estimator) {
    estimator_result_set_.reset(new ResultSet(ra_exe_unit.estimator, ExecutorDeviceType::CPU, 0, nullptr));
    out_vec.push_back(reinterpret_cast<int64_t*>(estimator_result_set_->getHostEstimatorBuffer()));
  } else {
    if (!is_group_by) {
      for (size_t i = 0; i < init_agg_vals.size(); ++i) {
        auto buff = new int64_t[num_out_frags];
        out_vec.push_back(static_cast<int64_t*>(buff));
      }
    }
  }

  CHECK_EQ(num_rows.size(), col_buffers.size());
  std::vector<int64_t> flatened_num_rows;
  OOM_TRACE_PUSH();
  for (auto& nums : num_rows) {
    flatened_num_rows.insert(flatened_num_rows.end(), nums.begin(), nums.end());
  }
  std::vector<uint64_t> flatened_frag_offsets;
  for (auto& offsets : frag_offsets) {
    flatened_frag_offsets.insert(flatened_frag_offsets.end(), offsets.begin(), offsets.end());
  }
  int64_t rowid_lookup_num_rows{*error_code ? *error_code + 1 : 0};
  auto num_rows_ptr = rowid_lookup_num_rows ? &rowid_lookup_num_rows : &flatened_num_rows[0];
  int32_t total_matched_init{0};

  std::vector<int64_t> cmpt_val_buff;
  if (is_group_by) {
    cmpt_val_buff = compact_init_vals(
        align_to_int64(query_mem_desc_.getColsSize()) / sizeof(int64_t), init_agg_vals, query_mem_desc_.agg_col_widths);
  }

  const int64_t* join_hash_tables_ptr = join_hash_tables.size() == 1
                                            ? reinterpret_cast<int64_t*>(join_hash_tables[0])
                                            : (join_hash_tables.size() > 1 ? &join_hash_tables[0] : nullptr);
  if (hoist_literals) {
    typedef void (*agg_query)(const int8_t*** col_buffers,
                              const uint32_t* num_fragments,
                              const uint32_t* frag_stride,
                              const int8_t* literals,
                              const int64_t* num_rows,
                              const uint64_t* frag_row_offsets,
                              const int32_t* max_matched,
                              int32_t* total_matched,
                              const int64_t* init_agg_value,
                              int64_t** out,
                              int64_t** out2,
                              int32_t* error_code,
                              const uint32_t* num_tables,
                              const int64_t* join_hash_tables_ptr);
    if (is_group_by) {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    &literal_buff[0],
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &cmpt_val_buff[0],
                                                    &group_by_buffers_[0],
                                                    small_group_by_buffers_ptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    } else {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    &literal_buff[0],
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &init_agg_vals[0],
                                                    &out_vec[0],
                                                    nullptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    }
  } else {
    typedef void (*agg_query)(const int8_t*** col_buffers,
                              const uint32_t* num_fragments,
                              const uint32_t* frag_stride,
                              const int64_t* num_rows,
                              const uint64_t* frag_row_offsets,
                              const int32_t* max_matched,
                              int32_t* total_matched,
                              const int64_t* init_agg_value,
                              int64_t** out,
                              int64_t** out2,
                              int32_t* error_code,
                              const uint32_t* num_tables,
                              const int64_t* join_hash_tables_ptr);
    if (is_group_by) {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &cmpt_val_buff[0],
                                                    &group_by_buffers_[0],
                                                    small_group_by_buffers_ptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    } else {
      OOM_TRACE_PUSH();
      reinterpret_cast<agg_query>(fn_ptrs[0].first)(multifrag_cols_ptr,
                                                    &num_fragments,
                                                    &frag_stride,
                                                    num_rows_ptr,
                                                    &flatened_frag_offsets[0],
                                                    &scan_limit,
                                                    &total_matched_init,
                                                    &init_agg_vals[0],
                                                    &out_vec[0],
                                                    nullptr,
                                                    error_code,
                                                    &num_tables,
                                                    join_hash_tables_ptr);
    }
  }

  if (ra_exe_unit.estimator) {
    return {};
  }

  if (rowid_lookup_num_rows && *error_code < 0) {
    *error_code = 0;
  }

  if (use_streaming_top_n(ra_exe_unit, query_mem_desc_)) {
    CHECK_EQ(group_by_buffers_.size(), size_t(1));
    const auto rows_copy = streaming_top_n::get_rows_copy_from_heaps(
        group_by_buffers_[0],
        query_mem_desc_.getBufferSizeBytes(ra_exe_unit, 1, ExecutorDeviceType::CPU),
        ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit,
        1);
    CHECK_EQ(rows_copy.size(), query_mem_desc_.entry_count * query_mem_desc_.getRowSize());
    memcpy(group_by_buffers_[0], &rows_copy[0], rows_copy.size());
  }

  return out_vec;
}

std::unique_ptr<QueryExecutionContext> QueryMemoryDescriptor::getQueryExecutionContext(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<int64_t>& init_agg_vals,
    const Executor* executor,
    const ExecutorDeviceType device_type,
    const int device_id,
    const std::vector<std::vector<const int8_t*>>& col_buffers,
    const std::vector<std::vector<const int8_t*>>& iter_buffers,
    const std::vector<std::vector<uint64_t>>& frag_offsets,
    std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
    const bool output_columnar,
    const bool sort_on_gpu,
    RenderInfo* render_info) const {
  return std::unique_ptr<QueryExecutionContext>(new QueryExecutionContext(ra_exe_unit,
                                                                          *this,
                                                                          init_agg_vals,
                                                                          executor,
                                                                          device_type,
                                                                          device_id,
                                                                          col_buffers,
                                                                          iter_buffers,
                                                                          frag_offsets,
                                                                          row_set_mem_owner,
                                                                          output_columnar,
                                                                          sort_on_gpu,
                                                                          render_info));
}

size_t QueryMemoryDescriptor::getColsSize() const {
  size_t total_bytes{0};
  for (size_t col_idx = 0; col_idx < agg_col_widths.size(); ++col_idx) {
    auto chosen_bytes = agg_col_widths[col_idx].compact;
    if (chosen_bytes == sizeof(int64_t)) {
      total_bytes = align_to_int64(total_bytes);
    }
    total_bytes += chosen_bytes;
  }
  return total_bytes;
}

size_t QueryMemoryDescriptor::getRowSize() const {
  CHECK(!output_columnar);
  size_t total_bytes{0};
  if (keyless_hash) {
    CHECK_EQ(size_t(1), group_col_widths.size());
  } else {
    total_bytes += group_col_widths.size() * getEffectiveKeyWidth();
    total_bytes = align_to_int64(total_bytes);
  }
  total_bytes += getColsSize();
  return align_to_int64(total_bytes);
}

size_t QueryMemoryDescriptor::getWarpCount() const {
  return (interleaved_bins_on_gpu ? executor_->warpSize() : 1);
}

size_t QueryMemoryDescriptor::getCompactByteWidth() const {
  if (agg_col_widths.empty()) {
    return 8;
  }
  size_t compact_width{0};
  for (const auto col_width : agg_col_widths) {
    if (col_width.compact != 0) {
      compact_width = col_width.compact;
      break;
    }
  }
  if (!compact_width) {
    return 0;
  }
  CHECK_GT(compact_width, size_t(0));
  for (const auto col_width : agg_col_widths) {
    if (col_width.compact == 0) {
      continue;
    }
    CHECK_EQ(col_width.compact, compact_width);
  }
  return compact_width;
}

// TODO(miyu): remove if unnecessary
bool QueryMemoryDescriptor::isCompactLayoutIsometric() const {
  if (agg_col_widths.empty()) {
    return true;
  }
  const auto compact_width = agg_col_widths.front().compact;
  for (const auto col_width : agg_col_widths) {
    if (col_width.compact != compact_width) {
      return false;
    }
  }
  return true;
}

size_t QueryMemoryDescriptor::getTotalBytesOfColumnarBuffers(const std::vector<ColWidths>& col_widths) const {
  CHECK(output_columnar);
  size_t total_bytes{0};
  const auto is_isometric = isCompactLayoutIsometric();
  for (size_t col_idx = 0; col_idx < col_widths.size(); ++col_idx) {
    total_bytes += col_widths[col_idx].compact * entry_count;
    if (!is_isometric) {
      total_bytes = align_to_int64(total_bytes);
    }
  }
  return align_to_int64(total_bytes);
}

size_t QueryMemoryDescriptor::getKeyOffInBytes(const size_t bin, const size_t key_idx) const {
  CHECK(!keyless_hash);
  if (output_columnar) {
    CHECK_EQ(size_t(0), key_idx);
    return bin * sizeof(int64_t);
  }

  CHECK_LT(key_idx, group_col_widths.size());
  auto offset = bin * getRowSize();
  CHECK_EQ(size_t(0), offset % sizeof(int64_t));
  offset += key_idx * getEffectiveKeyWidth();
  return offset;
}

size_t QueryMemoryDescriptor::getNextKeyOffInBytes(const size_t crt_idx) const {
  CHECK(!keyless_hash);
  CHECK_LT(crt_idx, group_col_widths.size());
  if (output_columnar) {
    CHECK_EQ(size_t(0), crt_idx);
  }
  return getEffectiveKeyWidth();
}

size_t QueryMemoryDescriptor::getColOnlyOffInBytes(const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  size_t offset{0};
  for (size_t index = 0; index < col_idx; ++index) {
    const auto chosen_bytes = agg_col_widths[index].compact;
    if (chosen_bytes == sizeof(int64_t)) {
      offset = align_to_int64(offset);
    }
    offset += chosen_bytes;
  }

  if (sizeof(int64_t) == agg_col_widths[col_idx].compact) {
    offset = align_to_int64(offset);
  }

  return offset;
}

size_t QueryMemoryDescriptor::getColOffInBytes(const size_t bin, const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  const auto warp_count = getWarpCount();
  if (output_columnar) {
    CHECK_LT(bin, entry_count);
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);
    size_t offset{0};
    const auto is_isometric = isCompactLayoutIsometric();
    if (!keyless_hash) {
      offset = sizeof(int64_t) * entry_count;
    }
    for (size_t index = 0; index < col_idx; ++index) {
      offset += agg_col_widths[index].compact * entry_count;
      if (!is_isometric) {
        offset = align_to_int64(offset);
      }
    }
    offset += bin * agg_col_widths[col_idx].compact;
    return offset;
  }

  auto offset = bin * warp_count * getRowSize();
  if (keyless_hash) {
    CHECK_EQ(size_t(1), group_col_widths.size());
  } else {
    offset += group_col_widths.size() * getEffectiveKeyWidth();
    offset = align_to_int64(offset);
  }
  offset += getColOnlyOffInBytes(col_idx);
  return offset;
}

size_t QueryMemoryDescriptor::getConsistColOffInBytes(const size_t bin, const size_t col_idx) const {
  CHECK(output_columnar && !agg_col_widths.empty());
  return (keyless_hash ? 0 : sizeof(int64_t) * entry_count) + (col_idx * entry_count + bin) * agg_col_widths[0].compact;
}

size_t QueryMemoryDescriptor::getColOffInBytesInNextBin(const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  auto warp_count = getWarpCount();
  if (output_columnar) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);
    return agg_col_widths[col_idx].compact;
  }

  return warp_count * getRowSize();
}

size_t QueryMemoryDescriptor::getNextColOffInBytes(const int8_t* col_ptr,
                                                   const size_t bin,
                                                   const size_t col_idx) const {
  CHECK_LT(col_idx, agg_col_widths.size());
  CHECK(!output_columnar || bin < entry_count);
  size_t offset{0};
  auto warp_count = getWarpCount();
  const auto chosen_bytes = agg_col_widths[col_idx].compact;
  if (col_idx + 1 == agg_col_widths.size()) {
    if (output_columnar) {
      return (entry_count - bin) * chosen_bytes;
    } else {
      return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
    }
  }

  const auto next_chosen_bytes = agg_col_widths[col_idx + 1].compact;
  if (output_columnar) {
    CHECK_EQ(size_t(1), group_col_widths.size());
    CHECK_EQ(size_t(1), warp_count);

    offset = entry_count * chosen_bytes;
    if (!isCompactLayoutIsometric()) {
      offset = align_to_int64(offset);
    }
    offset += bin * (next_chosen_bytes - chosen_bytes);
    return offset;
  }

  if (next_chosen_bytes == sizeof(int64_t)) {
    return static_cast<size_t>(align_to_int64(col_ptr + chosen_bytes) - col_ptr);
  } else {
    return chosen_bytes;
  }
}

size_t QueryMemoryDescriptor::getBufferSizeQuad(const ExecutorDeviceType device_type) const {
  const auto size_bytes = getBufferSizeBytes(device_type);
  CHECK_EQ(size_t(0), size_bytes % sizeof(int64_t));
  return getBufferSizeBytes(device_type) / sizeof(int64_t);
}

size_t QueryMemoryDescriptor::getSmallBufferSizeQuad() const {
  CHECK(!keyless_hash || entry_count_small == 0);
  return (group_col_widths.size() + agg_col_widths.size()) * entry_count_small;
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(const RelAlgExecutionUnit& ra_exe_unit,
                                                 const unsigned thread_count,
                                                 const ExecutorDeviceType device_type) const {
  if (use_streaming_top_n(ra_exe_unit, *this)) {
    const size_t n = ra_exe_unit.sort_info.offset + ra_exe_unit.sort_info.limit;
    return streaming_top_n::get_heap_size(getRowSize(), n, thread_count);
  }
  return getBufferSizeBytes(device_type);
}

size_t QueryMemoryDescriptor::getBufferSizeBytes(const ExecutorDeviceType device_type) const {
  if (keyless_hash) {
    CHECK_GE(group_col_widths.size(), size_t(1));
    auto total_bytes = align_to_int64(getColsSize());

    return (interleavedBins(device_type) ? executor_->warpSize() : 1) * entry_count * total_bytes;
  }

  size_t total_bytes{0};
  if (output_columnar) {
    total_bytes =
        sizeof(int64_t) * group_col_widths.size() * entry_count + getTotalBytesOfColumnarBuffers(agg_col_widths);
  } else {
    total_bytes = getRowSize() * entry_count;
  }

  return total_bytes;
}

size_t QueryMemoryDescriptor::getSmallBufferSizeBytes() const {
  return getSmallBufferSizeQuad() * sizeof(int64_t);
}

namespace {

int32_t get_agg_count(const std::vector<Analyzer::Expr*>& target_exprs) {
  int32_t agg_count{0};
  for (auto target_expr : target_exprs) {
    CHECK(target_expr);
    const auto agg_expr = dynamic_cast<Analyzer::AggExpr*>(target_expr);
    if (!agg_expr || agg_expr->get_aggtype() == kLAST_SAMPLE) {
      const auto& ti = target_expr->get_type_info();
      if (ti.is_string() && ti.get_compression() != kENCODING_DICT) {
        agg_count += 2;
      } else {
        ++agg_count;
      }
      continue;
    }
    if (agg_expr && agg_expr->get_aggtype() == kAVG) {
      agg_count += 2;
    } else {
      ++agg_count;
    }
  }
  return agg_count;
}

bool expr_is_rowid(const Analyzer::Expr* expr, const Catalog_Namespace::Catalog& cat) {
  const auto col = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (!col) {
    return false;
  }
  const auto cd = get_column_descriptor_maybe(col->get_column_id(), col->get_table_id(), cat);
  if (!cd || !cd->isVirtualCol) {
    return false;
  }
  CHECK_EQ("rowid", cd->columnName);
  return true;
}

bool has_count_distinct(const RelAlgExecutionUnit& ra_exe_unit) {
  for (const auto& target_expr : ra_exe_unit.target_exprs) {
    const auto agg_info = target_info(target_expr);
    if (agg_info.is_agg && is_distinct_target(agg_info)) {
      return true;
    }
  }
  return false;
}

}  // namespace

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getColRangeInfo() {
  // Use baseline layout more eagerly on the GPU if the query uses count distinct,
  // because our HyperLogLog implementation is 4x less memory efficient on GPU.
  // Technically, this only applies to APPROX_COUNT_DISTINCT, but in practice we
  // can expect this to be true anyway for grouped queries since the precise version
  // uses significantly more memory.
  const int64_t baseline_threshold = has_count_distinct(ra_exe_unit_)
                                         ? (device_type_ == ExecutorDeviceType::GPU ? (Executor::baseline_threshold / 4)
                                                                                    : Executor::baseline_threshold)
                                         : Executor::baseline_threshold;
  if (ra_exe_unit_.groupby_exprs.size() != 1) {
    try {
      checked_int64_t cardinality{1};
      bool has_nulls{false};
      for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
        auto col_range_info = getExprRangeInfo(groupby_expr.get());
        if (col_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange) {
          return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
        }
        auto crt_col_cardinality = getBucketedCardinality(col_range_info);
        CHECK_GE(crt_col_cardinality, 0);
        cardinality *= crt_col_cardinality;
        if (col_range_info.has_nulls) {
          has_nulls = true;
        }
      }
      // For zero or high cardinalities, use baseline layout.
      if (!cardinality || cardinality > baseline_threshold) {
        return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
      }
      return {GroupByColRangeType::MultiColPerfectHash, 0, int64_t(cardinality), 0, has_nulls};
    } catch (...) {  // overflow when computing cardinality
      return {GroupByColRangeType::MultiCol, 0, 0, 0, false};
    }
  }
  const auto col_range_info = getExprRangeInfo(ra_exe_unit_.groupby_exprs.front().get());
  if (!ra_exe_unit_.groupby_exprs.front()) {
    return col_range_info;
  }
  static const int64_t MAX_BUFFER_SIZE = 1 << 30;
  const int64_t col_count = ra_exe_unit_.groupby_exprs.size() + ra_exe_unit_.target_exprs.size();
  int64_t max_entry_count = MAX_BUFFER_SIZE / (col_count * sizeof(int64_t));
  if (has_count_distinct(ra_exe_unit_)) {
    max_entry_count = std::min(max_entry_count, baseline_threshold);
  }
  if ((!ra_exe_unit_.groupby_exprs.front()->get_type_info().is_string() &&
       !expr_is_rowid(ra_exe_unit_.groupby_exprs.front().get(), *executor_->catalog_)) &&
      col_range_info.max >= col_range_info.min + max_entry_count && !col_range_info.bucket) {
    return {GroupByColRangeType::MultiCol, col_range_info.min, col_range_info.max, 0, col_range_info.has_nulls};
  }
  return col_range_info;
}

GroupByAndAggregate::ColRangeInfo GroupByAndAggregate::getExprRangeInfo(const Analyzer::Expr* expr) const {
  if (!expr) {
    return {GroupByColRangeType::Projection, 0, 0, 0, false};
  }
  const int64_t guessed_range_max{255};  // TODO(alex): replace with educated guess

  const auto expr_range = getExpressionRange(redirect_expr(expr, ra_exe_unit_.input_col_descs).get(),
                                             query_infos_,
                                             executor_,
                                             boost::make_optional(ra_exe_unit_.simple_quals));
  switch (expr_range.getType()) {
    case ExpressionRangeType::Integer:
      return {GroupByColRangeType::OneColKnownRange,
              expr_range.getIntMin(),
              expr_range.getIntMax(),
              expr_range.getBucket(),
              expr_range.hasNulls()};
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double:
    case ExpressionRangeType::Invalid:
      return (g_cluster || g_use_result_set)
                 ? ColRangeInfo{GroupByColRangeType::MultiCol, 0, 0, 0, false}
                 : ColRangeInfo{GroupByColRangeType::OneColGuessedRange, 0, guessed_range_max, 0, false};
    default:
      CHECK(false);
  }
  CHECK(false);
  return {GroupByColRangeType::Scan, 0, 0, 0, false};
}

int64_t GroupByAndAggregate::getBucketedCardinality(const GroupByAndAggregate::ColRangeInfo& col_range_info) {
  auto crt_col_cardinality = col_range_info.max - col_range_info.min;
  if (col_range_info.bucket) {
    crt_col_cardinality /= col_range_info.bucket;
  }
  return crt_col_cardinality + (1 + (col_range_info.has_nulls ? 1 : 0));
}

#define LL_CONTEXT executor_->cgen_state_->context_
#define LL_BUILDER executor_->cgen_state_->ir_builder_
#define LL_BOOL(v) executor_->ll_bool(v)
#define LL_INT(v) executor_->ll_int(v)
#define LL_FP(v) executor_->ll_fp(v)
#define ROW_FUNC executor_->cgen_state_->row_func_

namespace {

bool many_entries(const int64_t max_val, const int64_t min_val, const int64_t bucket) {
  return max_val - min_val > 10000 * std::max(bucket, int64_t(1));
}

bool is_int_and_no_bigger_than(const SQLTypeInfo& ti, const size_t byte_width) {
  if (!ti.is_integer()) {
    return false;
  }
  return get_bit_width(ti) <= (byte_width * 8);
}

}  // namespace

GroupByAndAggregate::GroupByAndAggregate(Executor* executor,
                                         const ExecutorDeviceType device_type,
                                         const RelAlgExecutionUnit& ra_exe_unit,
                                         RenderInfo* render_info,
                                         const std::vector<InputTableInfo>& query_infos,
                                         std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                                         const size_t max_groups_buffer_entry_count,
                                         const size_t small_groups_buffer_entry_count,
                                         const int8_t crt_min_byte_width,
                                         const bool allow_multifrag,
                                         const bool output_columnar_hint)
    : executor_(executor),
      ra_exe_unit_(ra_exe_unit),
      query_infos_(query_infos),
      row_set_mem_owner_(row_set_mem_owner),
      device_type_(device_type) {
  for (const auto groupby_expr : ra_exe_unit.groupby_exprs) {
    if (!groupby_expr) {
      continue;
    }
    const auto& groupby_ti = groupby_expr->get_type_info();
    if (groupby_ti.is_string() && groupby_ti.get_compression() != kENCODING_DICT) {
      throw std::runtime_error("Cannot group by string columns which are not dictionary encoded.");
    }
    if (groupby_ti.is_array()) {
      throw std::runtime_error("Group by array not supported");
    }
    if (groupby_ti.is_geometry()) {
      throw std::runtime_error("Group by geometry not supported");
    }
  }
  const auto shard_count =
      device_type_ == ExecutorDeviceType::GPU ? shard_count_for_top_groups(ra_exe_unit_, *executor_->getCatalog()) : 0;
  bool sort_on_gpu_hint = device_type == ExecutorDeviceType::GPU && allow_multifrag &&
                          !ra_exe_unit.sort_info.order_entries.empty() &&
                          gpuCanHandleOrderEntries(ra_exe_unit.sort_info.order_entries) && !shard_count;
  // must_use_baseline_sort is true iff we'd sort on GPU with the old algorithm
  // but the total output buffer size would be too big or it's a sharded top query.
  // For the sake of managing risk, use the new result set way very selectively for
  // this case only (alongside the baseline layout we've enabled for a while now).
  bool must_use_baseline_sort = shard_count;
  while (true) {
    initQueryMemoryDescriptor(allow_multifrag,
                              max_groups_buffer_entry_count,
                              small_groups_buffer_entry_count,
                              crt_min_byte_width,
                              sort_on_gpu_hint,
                              render_info,
                              must_use_baseline_sort);
    if (device_type != ExecutorDeviceType::GPU) {
      // TODO(miyu): remove w/ interleaving
      query_mem_desc_.interleaved_bins_on_gpu = false;
    }
    query_mem_desc_.sort_on_gpu_ =
        sort_on_gpu_hint && query_mem_desc_.canOutputColumnar() && !query_mem_desc_.keyless_hash;
    query_mem_desc_.is_sort_plan = !ra_exe_unit.sort_info.order_entries.empty() && !query_mem_desc_.sort_on_gpu_;
    output_columnar_ = (output_columnar_hint && query_mem_desc_.canOutputColumnar()) || query_mem_desc_.sortOnGpu();
    query_mem_desc_.output_columnar = output_columnar_;
    if (query_mem_desc_.sortOnGpu() &&
        (query_mem_desc_.getBufferSizeBytes(device_type_) +
         align_to_int64(query_mem_desc_.entry_count * sizeof(int32_t))) > 2 * 1024 * 1024 * 1024L) {
      must_use_baseline_sort = true;
      sort_on_gpu_hint = false;
    } else {
      break;
    }
  }
}

int8_t pick_target_compact_width(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<InputTableInfo>& query_infos,
                                 const int8_t crt_min_byte_width) {
  if (g_bigint_count) {
    return sizeof(int64_t);
  }
  int8_t compact_width{0};
  auto col_it = ra_exe_unit.input_col_descs.begin();
  int unnest_array_col_id{std::numeric_limits<int>::min()};
  for (const auto groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto uoper = dynamic_cast<Analyzer::UOper*>(groupby_expr.get());
    if (uoper && uoper->get_optype() == kUNNEST) {
      const auto& arg_ti = uoper->get_operand()->get_type_info();
      CHECK(arg_ti.is_array());
      const auto& elem_ti = arg_ti.get_elem_type();
      if (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT) {
        unnest_array_col_id = (*col_it)->getColId();
      } else {
        compact_width = crt_min_byte_width;
        break;
      }
    }
    ++col_it;
  }
  if (!compact_width && (ra_exe_unit.groupby_exprs.size() != 1 || !ra_exe_unit.groupby_exprs.front())) {
    compact_width = crt_min_byte_width;
  }
  if (!compact_width) {
    col_it = ra_exe_unit.input_col_descs.begin();
    std::advance(col_it, ra_exe_unit.groupby_exprs.size());
    for (const auto target : ra_exe_unit.target_exprs) {
      const auto& ti = target->get_type_info();
      const auto agg = dynamic_cast<const Analyzer::AggExpr*>(target);
      if (agg && agg->get_arg()) {
        compact_width = crt_min_byte_width;
        break;
      }

      if (agg) {
        CHECK_EQ(kCOUNT, agg->get_aggtype());
        CHECK(!agg->get_is_distinct());
        ++col_it;
        continue;
      }

      if (is_int_and_no_bigger_than(ti, 4) || (ti.is_string() && ti.get_compression() == kENCODING_DICT)) {
        ++col_it;
        continue;
      }

      const auto uoper = dynamic_cast<Analyzer::UOper*>(target);
      if (uoper && uoper->get_optype() == kUNNEST && (*col_it)->getColId() == unnest_array_col_id) {
        const auto arg_ti = uoper->get_operand()->get_type_info();
        CHECK(arg_ti.is_array());
        const auto& elem_ti = arg_ti.get_elem_type();
        if (elem_ti.is_string() && elem_ti.get_compression() == kENCODING_DICT) {
          ++col_it;
          continue;
        }
      }

      compact_width = crt_min_byte_width;
      break;
    }
  }
  if (!compact_width) {
    size_t total_tuples{0};
    for (const auto& qi : query_infos) {
      total_tuples += qi.info.getNumTuples();
    }
    return total_tuples <= static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
                   unnest_array_col_id != std::numeric_limits<int>::min()
               ? 4
               : crt_min_byte_width;
  } else {
    // TODO(miyu): relax this condition to allow more cases just w/o padding
    for (auto wid : get_col_byte_widths(ra_exe_unit.target_exprs, {})) {
      compact_width = std::max(compact_width, wid);
    }
    return compact_width;
  }
}

namespace {

#ifdef ENABLE_KEY_COMPACTION
int8_t pick_baseline_key_component_width(const ExpressionRange& range) {
  if (range.getType() == ExpressionRangeType::Invalid) {
    return sizeof(int64_t);
  }
  switch (range.getType()) {
    case ExpressionRangeType::Integer:
      return range.getIntMax() < EMPTY_KEY_32 - 1 ? sizeof(int32_t) : sizeof(int64_t);
    case ExpressionRangeType::Float:
    case ExpressionRangeType::Double:
      return sizeof(int64_t);  // No compaction for floating point yet.
    default:
      CHECK(false);
  }
  return sizeof(int64_t);
}

// TODO(miyu): make sure following setting of compact width is correct in all cases.
int8_t pick_baseline_key_width(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<InputTableInfo>& query_infos,
                               const Executor* executor) {
  int8_t compact_width{4};
  for (const auto groupby_expr : ra_exe_unit.groupby_exprs) {
    const auto actual_expr = redirect_expr(groupby_expr.get(), ra_exe_unit.input_col_descs);
    const auto expr_range = getExpressionRange(actual_expr.get(), query_infos, executor);
    compact_width = std::max(compact_width, pick_baseline_key_component_width(expr_range));
  }
  return compact_width;
}
#endif

std::vector<ssize_t> target_expr_group_by_indices(const std::list<std::shared_ptr<Analyzer::Expr>>& groupby_exprs,
                                                  const std::vector<Analyzer::Expr*>& target_exprs) {
  std::vector<ssize_t> indices(target_exprs.size(), -1);
  for (size_t target_idx = 0; target_idx < target_exprs.size(); ++target_idx) {
    const auto target_expr = target_exprs[target_idx];
    if (dynamic_cast<const Analyzer::AggExpr*>(target_expr)) {
      continue;
    }
    size_t group_idx = 0;
    for (const auto groupby_expr : groupby_exprs) {
      if (*target_expr == *groupby_expr) {
        indices[target_idx] = group_idx;
        break;
      }
      ++group_idx;
    }
  }
  return indices;
}

}  // namespace

int64_t GroupByAndAggregate::getShardedTopBucket(const ColRangeInfo& col_range_info, const size_t shard_count) const {
  int device_count{0};
  if (device_type_ == ExecutorDeviceType::GPU) {
    device_count = executor_->getCatalog()->get_dataMgr().cudaMgr_->getDeviceCount();
    CHECK_GT(device_count, 0);
  }

  int64_t bucket{col_range_info.bucket};

  if (shard_count) {
    CHECK(!col_range_info.bucket);
    if (static_cast<size_t>(device_count) <= shard_count && g_leaf_count) {
      bucket = shard_count * g_leaf_count;
    } else {
      bucket = device_count;
    }
  }

  return bucket;
}

namespace {

class UsedColumnsVisitor : public ScalarExprVisitor<std::unordered_set<int>> {
 protected:
  virtual std::unordered_set<int> visitColumnVar(const Analyzer::ColumnVar* column) const override {
    return {column->get_column_id()};
  }

  virtual std::unordered_set<int> aggregateResult(const std::unordered_set<int>& aggregate,
                                                  const std::unordered_set<int>& next_result) const override {
    auto result = aggregate;
    result.insert(next_result.begin(), next_result.end());
    return result;
  }
};

std::vector<ssize_t> target_expr_proj_indices(const RelAlgExecutionUnit& ra_exe_unit,
                                              const Catalog_Namespace::Catalog& cat) {
  if (ra_exe_unit.input_descs.size() > 1 || !ra_exe_unit.sort_info.order_entries.empty()) {
    return {};
  }
  std::vector<ssize_t> target_indices(ra_exe_unit.target_exprs.size(), -1);
  UsedColumnsVisitor columns_visitor;
  std::unordered_set<int> used_columns;
  for (const auto& simple_qual : ra_exe_unit.simple_quals) {
    const auto crt_used_columns = columns_visitor.visit(simple_qual.get());
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (const auto& qual : ra_exe_unit.quals) {
    const auto crt_used_columns = columns_visitor.visit(qual.get());
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (const auto& target : ra_exe_unit.target_exprs) {
    const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target);
    if (col_var) {
      const auto cd = get_column_descriptor_maybe(col_var->get_column_id(), col_var->get_table_id(), cat);
      if (!cd || !cd->isVirtualCol) {
        continue;
      }
    }
    const auto crt_used_columns = columns_visitor.visit(target);
    used_columns.insert(crt_used_columns.begin(), crt_used_columns.end());
  }
  for (size_t target_idx = 0; target_idx < ra_exe_unit.target_exprs.size(); ++target_idx) {
    const auto target_expr = ra_exe_unit.target_exprs[target_idx];
    CHECK(target_expr);
    const auto& ti = target_expr->get_type_info();
    const bool is_real_str_or_array = (ti.is_string() && ti.get_compression() == kENCODING_NONE) || ti.is_array();
    if (is_real_str_or_array) {
      continue;
    }
    if (ti.is_geometry()) {
      // TODO(adb): Ideally we could determine which physical columns are required for a given query and fetch only
      // those. For now, we bail on the memory optimization, since it is possible that adding the physical columns could
      // have unintended consequences further down the execution path.
      return {};
    }
    const auto col_var = dynamic_cast<const Analyzer::ColumnVar*>(target_expr);
    if (!col_var) {
      continue;
    }
    if (!is_real_str_or_array && used_columns.find(col_var->get_column_id()) == used_columns.end()) {
      target_indices[target_idx] = 0;
    }
  }
  return target_indices;
}

}  // namespace

void GroupByAndAggregate::initQueryMemoryDescriptor(const bool allow_multifrag,
                                                    const size_t max_groups_buffer_entry_count,
                                                    const size_t small_groups_buffer_entry_count,
                                                    const int8_t crt_min_byte_width,
                                                    const bool sort_on_gpu_hint,
                                                    RenderInfo* render_info,
                                                    const bool must_use_baseline_sort) {
  addTransientStringLiterals();

  const auto count_distinct_descriptors = initCountDistinctDescriptors();

  std::vector<ColWidths> agg_col_widths;
  const auto min_byte_width = pick_target_compact_width(ra_exe_unit_, query_infos_, crt_min_byte_width);
  for (auto wid : get_col_byte_widths(ra_exe_unit_.target_exprs, {})) {
    agg_col_widths.push_back({wid, static_cast<int8_t>(compact_byte_width(wid, min_byte_width))});
  }
  auto group_col_widths = get_col_byte_widths(ra_exe_unit_.groupby_exprs, {});

  const bool is_group_by{!group_col_widths.empty()};
  if (!is_group_by) {
    CHECK(!must_use_baseline_sort);
    CHECK(!render_info || !render_info->isPotentialInSituRender());
    query_mem_desc_ = {executor_,
                       allow_multifrag,
                       ra_exe_unit_.estimator ? GroupByColRangeType::Estimator : GroupByColRangeType::Scan,
                       false,
                       false,
                       -1,
                       0,
                       group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
                       0,
#endif
                       agg_col_widths,
                       {},
                       1,
                       0,
                       0,
                       0,
                       0,
                       false,
                       GroupByMemSharing::Private,
                       count_distinct_descriptors,
                       false,
                       false,
                       false,
                       false,
                       {},
                       {},
                       false};
    return;
  }

  auto col_range_info_nosharding = getColRangeInfo();

  const auto shard_count =
      device_type_ == ExecutorDeviceType::GPU ? shard_count_for_top_groups(ra_exe_unit_, *executor_->getCatalog()) : 0;

  const auto col_range_info = ColRangeInfo{col_range_info_nosharding.hash_type_,
                                           col_range_info_nosharding.min,
                                           col_range_info_nosharding.max,
                                           getShardedTopBucket(col_range_info_nosharding, shard_count),
                                           col_range_info_nosharding.has_nulls};

  // For timestamp operations, adjust tollerance as integer overflows for nanosecs precision.
  const int64_t tol = 130000000 * std::max(col_range_info.bucket, int64_t(1)) < 0
                          ? 130000 * std::max(col_range_info.bucket, int64_t(1))
                          : 130000000 * std::max(col_range_info.bucket, int64_t(1));

  if (g_enable_watchdog &&
      ((col_range_info.hash_type_ == GroupByColRangeType::MultiCol && max_groups_buffer_entry_count > 120000000) ||
       (col_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange &&
        col_range_info.max - col_range_info.min > tol))) {
    throw WatchdogException("Query would use too much memory");
  }

  switch (col_range_info.hash_type_) {
    case GroupByColRangeType::OneColKnownRange: {
      CHECK(!render_info || !render_info->isPotentialInSituRender());
      const auto redirected_targets = redirect_exprs(ra_exe_unit_.target_exprs, ra_exe_unit_.input_col_descs);
      const auto keyless_info = getKeylessInfo(get_exprs_not_owned(redirected_targets), is_group_by);
      bool keyless =
          (!sort_on_gpu_hint || !many_entries(col_range_info.max, col_range_info.min, col_range_info.bucket)) &&
          !col_range_info.bucket && !must_use_baseline_sort && keyless_info.keyless;
      size_t bin_count = std::max(getBucketedCardinality(col_range_info), int64_t(1));
      const size_t interleaved_max_threshold{512};

      size_t gpu_smem_max_threshold{0};
      if (device_type_ == ExecutorDeviceType::GPU) {
        const auto cuda_manager = executor_->getCatalog()->get_dataMgr().cudaMgr_;
        CHECK(cuda_manager);
        /*
         *  We only use shared memory strategy if GPU hardware provides native shared memory atomics support.
         *  From CUDA Toolkit documentation: https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html#atomic-ops
         *  "Like Maxwell, Pascal [and Volta] provides native shared memory atomic operations for 32-bit integer
         *arithmetic, along with native 32 or 64-bit compare-and-swap (CAS)."
         *
         **/
        if (cuda_manager->isArchMaxwellOrLaterForAll()) {
          // TODO(Saman): threshold should be eventually set as an optimized policy per architecture.
          gpu_smem_max_threshold = std::min((cuda_manager->isArchVoltaForAll()) ? 4095LU : 2047LU,
                                            (cuda_manager->maxSharedMemoryForAll / sizeof(int64_t) - 1));
        }
      }

      const auto group_expr = ra_exe_unit_.groupby_exprs.front().get();
      bool shared_mem_for_group_by = g_enable_smem_group_by && keyless && keyless_info.shared_mem_support &&
                                     (bin_count <= gpu_smem_max_threshold) &&
                                     (supportedExprForGpuSharedMemUsage(group_expr)) &&
                                     countDescriptorsLogicallyEmpty(count_distinct_descriptors);

      // No need to interleave results if we use shared memory.
      bool interleaved_bins = !shared_mem_for_group_by && keyless && (bin_count <= interleaved_max_threshold) &&
                              countDescriptorsLogicallyEmpty(count_distinct_descriptors);
      std::vector<ssize_t> target_group_by_indices;
      if (must_use_baseline_sort) {
        target_group_by_indices = target_expr_group_by_indices(ra_exe_unit_.groupby_exprs, ra_exe_unit_.target_exprs);
        agg_col_widths.clear();
        for (auto wid : get_col_byte_widths(ra_exe_unit_.target_exprs, target_group_by_indices)) {
          agg_col_widths.push_back({wid, static_cast<int8_t>(wid ? 8 : 0)});
        }
      }
      query_mem_desc_ = {
          executor_,
          allow_multifrag,
          col_range_info.hash_type_,
          keyless,
          interleaved_bins,
          keyless_info.target_index,
          keyless_info.init_val,
          group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
          0,
#endif
          agg_col_widths,
          target_group_by_indices,
          bin_count,
          0,
          col_range_info.min,
          col_range_info.max,
          col_range_info.bucket,
          col_range_info.has_nulls,
          shared_mem_for_group_by ? GroupByMemSharing::SharedForKeylessOneColumnKnownRange : GroupByMemSharing::Shared,
          count_distinct_descriptors,
          false,
          false,
          false,
          false,
          {},
          {},
          must_use_baseline_sort};
      // TODO(Saman): should remove this after implementing shared memory path completely through codegen
      if (shared_mem_for_group_by && (query_mem_desc_.getRowSize() > sizeof(int64_t))) {
        // We should not use the current shared memory path if more than 8 bytes per group is required
        query_mem_desc_.sharing = GroupByMemSharing::Shared;  // disable the new shared memory path
        query_mem_desc_.interleaved_bins_on_gpu = keyless && (bin_count <= interleaved_max_threshold) &&
                                                  countDescriptorsLogicallyEmpty(count_distinct_descriptors);
      }
      return;
    }
    case GroupByColRangeType::OneColGuessedRange: {
      CHECK(!must_use_baseline_sort);
      auto doRender = render_info && render_info->isPotentialInSituRender();
      query_mem_desc_ = {
          executor_,
          allow_multifrag,
          col_range_info.hash_type_,
          false,
          false,
          -1,
          0,
          group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
          0,
#endif
          agg_col_widths,
          {},
          max_groups_buffer_entry_count,
          doRender ? render_info->render_small_groups_buffer_entry_count : small_groups_buffer_entry_count,
          col_range_info.min,
          col_range_info.max,
          0,
          col_range_info.has_nulls,
          GroupByMemSharing::Shared,
          count_distinct_descriptors,
          false,
          false,
          false,
          doRender,
          {},
          {},
          false};
      return;
    }
    case GroupByColRangeType::MultiCol: {
      CHECK(!render_info || !render_info->isPotentialInSituRender());
      const auto target_group_by_indices =
          target_expr_group_by_indices(ra_exe_unit_.groupby_exprs, ra_exe_unit_.target_exprs);
      agg_col_widths.clear();
      for (auto wid : get_col_byte_widths(ra_exe_unit_.target_exprs, target_group_by_indices)) {
        // Baseline layout goes through new result set and ResultSetStorage::initializeRowWise
        // assumes everything is padded to 8 bytes, make it so.
        agg_col_widths.push_back({wid, static_cast<int8_t>(wid ? 8 : 0)});
      }
      const auto entries_per_shard =
          shard_count ? (max_groups_buffer_entry_count + shard_count - 1) / shard_count : max_groups_buffer_entry_count;
      query_mem_desc_ = {executor_,
                         allow_multifrag,
                         col_range_info.hash_type_,
                         false,
                         false,
                         -1,
                         0,
                         group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
                         pick_baseline_key_width(ra_exe_unit_, query_infos_, executor_),
#endif
                         agg_col_widths,
                         target_group_by_indices,
                         entries_per_shard,
                         0,
                         0,
                         0,
                         0,
                         false,
                         GroupByMemSharing::Shared,
                         count_distinct_descriptors,
                         false,
                         false,
                         false,
                         false,
                         {},
                         {},
                         false};
      return;
    }
    case GroupByColRangeType::Projection: {
      CHECK(!must_use_baseline_sort);
      size_t group_slots =
          ra_exe_unit_.scan_limit ? static_cast<size_t>(ra_exe_unit_.scan_limit) : max_groups_buffer_entry_count;
      const auto catalog = executor_->getCatalog();
      CHECK(catalog);
      const auto target_indices = executor_->plan_state_->allow_lazy_fetch_
                                      ? target_expr_proj_indices(ra_exe_unit_, *catalog)
                                      : std::vector<ssize_t>{};
      agg_col_widths.clear();
      for (auto wid : get_col_byte_widths(ra_exe_unit_.target_exprs, target_indices)) {
        // Baseline layout goes through new result set and ResultSetStorage::initializeRowWise
        // assumes everything is padded to 8 bytes, make it so.
        agg_col_widths.push_back({wid, static_cast<int8_t>(wid ? 8 : 0)});
      }
      query_mem_desc_ = {executor_,
                         allow_multifrag,
                         col_range_info.hash_type_,
                         false,
                         false,
                         -1,
                         0,
                         group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
                         0,
#endif
                         agg_col_widths,
                         target_indices,
                         group_slots,
                         0,
                         col_range_info.min,
                         col_range_info.max,
                         0,
                         col_range_info.has_nulls,
                         GroupByMemSharing::Shared,
                         count_distinct_descriptors,
                         false,
                         false,
                         false,
                         render_info && render_info->isPotentialInSituRender(),
                         {},
                         {},
                         false};
      if (use_streaming_top_n(ra_exe_unit_, query_mem_desc_)) {
        query_mem_desc_.entry_count = ra_exe_unit_.sort_info.offset + ra_exe_unit_.sort_info.limit;
      }

      return;
    }
    case GroupByColRangeType::MultiColPerfectHash: {
      CHECK(!render_info || !render_info->isPotentialInSituRender());
      query_mem_desc_ = {executor_,
                         allow_multifrag,
                         col_range_info.hash_type_,
                         false,
                         false,
                         -1,
                         0,
                         group_col_widths,
#ifdef ENABLE_KEY_COMPACTION
                         0,
#endif
                         agg_col_widths,
                         {},
                         static_cast<size_t>(col_range_info.max),
                         0,
                         col_range_info.min,
                         col_range_info.max,
                         0,
                         col_range_info.has_nulls,
                         GroupByMemSharing::Shared,
                         count_distinct_descriptors,
                         false,
                         false,
                         false,
                         false,
                         {},
                         {},
                         false};
      return;
    }
    default:
      CHECK(false);
  }
  CHECK(false);
  return;
}

void GroupByAndAggregate::addTransientStringLiterals() {
  addTransientStringLiterals(ra_exe_unit_, executor_, row_set_mem_owner_);
}

namespace {

void add_transient_string_literals_for_expression(const Analyzer::Expr* expr,
                                                  Executor* executor,
                                                  std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  if (!expr) {
    return;
  }
  const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  const auto& expr_ti = expr->get_type_info();
  if (cast_expr && cast_expr->get_optype() == kCAST && expr_ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, expr_ti.get_compression());
    auto sdp = executor->getStringDictionaryProxy(expr_ti.get_comp_param(), row_set_mem_owner, true);
    CHECK(sdp);
    const auto str_lit_expr = dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand());
    if (str_lit_expr && str_lit_expr->get_constval().stringval) {
      sdp->getOrAddTransient(*str_lit_expr->get_constval().stringval);
    }
    return;
  }
  const auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (!case_expr) {
    return;
  }
  Analyzer::DomainSet domain_set;
  case_expr->get_domain(domain_set);
  if (domain_set.empty()) {
    return;
  }
  if (expr_ti.is_string()) {
    CHECK_EQ(kENCODING_DICT, expr_ti.get_compression());
    auto sdp = executor->getStringDictionaryProxy(expr_ti.get_comp_param(), row_set_mem_owner, true);
    CHECK(sdp);
    for (const auto domain_expr : domain_set) {
      const auto cast_expr = dynamic_cast<const Analyzer::UOper*>(domain_expr);
      const auto str_lit_expr = cast_expr && cast_expr->get_optype() == kCAST
                                    ? dynamic_cast<const Analyzer::Constant*>(cast_expr->get_operand())
                                    : dynamic_cast<const Analyzer::Constant*>(domain_expr);
      if (str_lit_expr && str_lit_expr->get_constval().stringval) {
        sdp->getOrAddTransient(*str_lit_expr->get_constval().stringval);
      }
    }
  }
}

}  // namespace

void GroupByAndAggregate::addTransientStringLiterals(const RelAlgExecutionUnit& ra_exe_unit,
                                                     Executor* executor,
                                                     std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner) {
  for (const auto group_expr : ra_exe_unit.groupby_exprs) {
    add_transient_string_literals_for_expression(group_expr.get(), executor, row_set_mem_owner);
  }
  for (const auto target_expr : ra_exe_unit.target_exprs) {
    const auto& target_type = target_expr->get_type_info();
    if (target_type.is_string() && target_type.get_compression() != kENCODING_DICT) {
      continue;
    }
    const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
    if (agg_expr) {
      if (agg_expr->get_aggtype() == kLAST_SAMPLE) {
        add_transient_string_literals_for_expression(agg_expr->get_arg(), executor, row_set_mem_owner);
      }
    } else {
      add_transient_string_literals_for_expression(target_expr, executor, row_set_mem_owner);
    }
  }
  row_set_mem_owner->addLiteralStringDictProxy(executor->lit_str_dict_proxy_);
}

CountDistinctDescriptors GroupByAndAggregate::initCountDistinctDescriptors() {
  CountDistinctDescriptors count_distinct_descriptors;
  for (const auto target_expr : ra_exe_unit_.target_exprs) {
    auto agg_info = target_info(target_expr);
    if (is_distinct_target(agg_info)) {
      CHECK(agg_info.is_agg);
      CHECK(agg_info.agg_kind == kCOUNT || agg_info.agg_kind == kAPPROX_COUNT_DISTINCT);
      const auto agg_expr = static_cast<const Analyzer::AggExpr*>(target_expr);
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_string() && arg_ti.get_compression() != kENCODING_DICT) {
        throw std::runtime_error("Strings must be dictionary-encoded for COUNT(DISTINCT).");
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT && arg_ti.is_array()) {
        throw std::runtime_error("APPROX_COUNT_DISTINCT on arrays not supported yet");
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT && arg_ti.is_geometry()) {
        throw std::runtime_error("APPROX_COUNT_DISTINCT on geometry columns not supported");
      }
      if (agg_info.is_distinct && arg_ti.is_geometry()) {
        throw std::runtime_error("COUNT DISTINCT on geometry columns not supported");
      }
      GroupByAndAggregate::ColRangeInfo no_range_info{GroupByColRangeType::OneColGuessedRange, 0, 0, 0, false};
      auto arg_range_info = arg_ti.is_fp() ? no_range_info : getExprRangeInfo(agg_expr->get_arg());
      CountDistinctImplType count_distinct_impl_type{CountDistinctImplType::StdSet};
      int64_t bitmap_sz_bits{0};
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
        const auto error_rate = agg_expr->get_error_rate();
        if (error_rate) {
          CHECK(error_rate->get_type_info().get_type() == kSMALLINT);
          CHECK_GE(error_rate->get_constval().smallintval, 1);
          bitmap_sz_bits = hll_size_for_rate(error_rate->get_constval().smallintval);
        } else {
          bitmap_sz_bits = g_hll_precision_bits;
        }
      }
      if (arg_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange &&
          !(arg_ti.is_array() || arg_ti.is_geometry())) {  // TODO(alex): allow bitmap implementation for arrays
        if (arg_range_info.isEmpty()) {
          count_distinct_descriptors.emplace_back(CountDistinctDescriptor{
              CountDistinctImplType::Bitmap, 0, 64, agg_info.agg_kind == kAPPROX_COUNT_DISTINCT, device_type_, 1});
          continue;
        }
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
        if (agg_info.agg_kind == kCOUNT) {
          bitmap_sz_bits = arg_range_info.max - arg_range_info.min + 1;
          const int64_t MAX_BITMAP_BITS{8 * 1000 * 1000 * 1000L};
          if (bitmap_sz_bits <= 0 || bitmap_sz_bits > MAX_BITMAP_BITS) {
            count_distinct_impl_type = CountDistinctImplType::StdSet;
          }
        }
      }
      if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT && count_distinct_impl_type == CountDistinctImplType::StdSet &&
          !(arg_ti.is_array() || arg_ti.is_geometry())) {
        count_distinct_impl_type = CountDistinctImplType::Bitmap;
      }
      if (g_enable_watchdog && count_distinct_impl_type == CountDistinctImplType::StdSet) {
        throw WatchdogException("Cannot use a fast path for COUNT distinct");
      }
      const auto sub_bitmap_count = get_count_distinct_sub_bitmap_count(bitmap_sz_bits, ra_exe_unit_, device_type_);
      count_distinct_descriptors.emplace_back(CountDistinctDescriptor{count_distinct_impl_type,
                                                                      arg_range_info.min,
                                                                      bitmap_sz_bits,
                                                                      agg_info.agg_kind == kAPPROX_COUNT_DISTINCT,
                                                                      device_type_,
                                                                      sub_bitmap_count});
    } else {
      count_distinct_descriptors.emplace_back(
          CountDistinctDescriptor{CountDistinctImplType::Invalid, 0, 0, false, device_type_, 0});
    }
  }
  return count_distinct_descriptors;
}

const QueryMemoryDescriptor& GroupByAndAggregate::getQueryMemoryDescriptor() const {
  return query_mem_desc_;
}

bool GroupByAndAggregate::outputColumnar() const {
  return output_columnar_;
}

GroupByAndAggregate::KeylessInfo GroupByAndAggregate::getKeylessInfo(
    const std::vector<Analyzer::Expr*>& target_expr_list,
    const bool is_group_by) const {
  bool keyless{true}, found{false}, shared_mem_support{false}, shared_mem_valid_data_type{true};
  /* Currently support shared memory usage for a limited subset of possible aggregate operations. shared_mem_support and
   * shared_mem_valid_data_type are declared to ensure such support. */
  int32_t num_agg_expr{0};  // used for shared memory support on the GPU
  int32_t index{0};
  int64_t init_val{0};
  for (const auto target_expr : target_expr_list) {
    const auto agg_info = target_info(target_expr);
    const auto& chosen_type = get_compact_type(agg_info);
    // TODO(Saman): should be eventually removed, once I make sure what data types can be used in this shared memory
    // setting.

    shared_mem_valid_data_type = shared_mem_valid_data_type && supportedTypeForGpuSharedMemUsage(chosen_type);

    if (agg_info.is_agg)
      num_agg_expr++;
    if (!found && agg_info.is_agg && !is_distinct_target(agg_info)) {
      auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
      CHECK(agg_expr);
      const auto arg_expr = agg_arg(target_expr);
      const bool float_argument_input = takes_float_argument(agg_info);
      switch (agg_info.agg_kind) {
        case kAVG:
          ++index;
          if (arg_expr && !arg_expr->get_type_info().get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid || expr_range_info.hasNulls()) {
              break;
            }
          }
          init_val = 0;
          found = true;
          break;
        case kCOUNT:
          if (arg_expr && !arg_expr->get_type_info().get_notnull()) {
            const auto& arg_ti = arg_expr->get_type_info();
            if (arg_ti.is_string() && arg_ti.get_compression() == kENCODING_NONE) {
              break;
            }
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            if (expr_range_info.getType() == ExpressionRangeType::Invalid || expr_range_info.hasNulls()) {
              break;
            }
          }
          init_val = 0;
          found = true;
          if (!agg_info.skip_null_val)
            shared_mem_support = true;  // currently just support 8 bytes per group
          break;
        case kSUM: {
          auto arg_ti = arg_expr->get_type_info();
          if (constrained_not_null(arg_expr, ra_exe_unit_.quals)) {
            arg_ti.set_notnull(true);
          }
          if (!arg_ti.get_notnull()) {
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            if (expr_range_info.getType() != ExpressionRangeType::Invalid && !expr_range_info.hasNulls()) {
              init_val =
                  get_agg_initial_val(agg_info.agg_kind,
                                      arg_ti,
                                      is_group_by || float_argument_input,
                                      float_argument_input ? sizeof(float) : query_mem_desc_.getCompactByteWidth());
              found = true;
            }
          } else {
            init_val = 0;
            auto expr_range_info = getExpressionRange(arg_expr, query_infos_, executor_);
            switch (expr_range_info.getType()) {
              case ExpressionRangeType::Float:
              case ExpressionRangeType::Double:
                if (expr_range_info.getFpMax() < 0 || expr_range_info.getFpMin() > 0) {
                  found = true;
                }
                break;
              case ExpressionRangeType::Integer:
                if (expr_range_info.getIntMax() < 0 || expr_range_info.getIntMin() > 0) {
                  found = true;
                }
                break;
              default:
                break;
            }
          }
          break;
        }
        case kMIN: {
          CHECK(agg_expr && agg_expr->get_arg());
          const auto& arg_ti = agg_expr->get_arg()->get_type_info();
          if (arg_ti.is_string() || arg_ti.is_array()) {
            break;
          }
          auto expr_range_info = getExpressionRange(agg_expr->get_arg(), query_infos_, executor_);
          auto init_max =
              get_agg_initial_val(agg_info.agg_kind,
                                  chosen_type,
                                  is_group_by || float_argument_input,
                                  float_argument_input ? sizeof(float) : query_mem_desc_.getCompactByteWidth());
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              init_val = init_max;
              auto double_max = *reinterpret_cast<const double*>(may_alias_ptr(&init_max));
              if (expr_range_info.getFpMax() < double_max) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              init_val = init_max;
              if (expr_range_info.getIntMax() < init_max) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        case kMAX: {
          CHECK(agg_expr && agg_expr->get_arg());
          const auto& arg_ti = agg_expr->get_arg()->get_type_info();
          if (arg_ti.is_string() || arg_ti.is_array()) {
            break;
          }
          auto expr_range_info = getExpressionRange(agg_expr->get_arg(), query_infos_, executor_);
          auto init_min =
              get_agg_initial_val(agg_info.agg_kind,
                                  chosen_type,
                                  is_group_by || float_argument_input,
                                  float_argument_input ? sizeof(float) : query_mem_desc_.getCompactByteWidth());
          switch (expr_range_info.getType()) {
            case ExpressionRangeType::Float:
            case ExpressionRangeType::Double: {
              init_val = init_min;
              auto double_min = *reinterpret_cast<const double*>(may_alias_ptr(&init_min));
              if (expr_range_info.getFpMin() > double_min) {
                found = true;
              }
              break;
            }
            case ExpressionRangeType::Integer:
              init_val = init_min;
              if (expr_range_info.getIntMin() > init_min) {
                found = true;
              }
              break;
            default:
              break;
          }
          break;
        }
        default:
          keyless = false;
          break;
      }
    }
    if (!keyless) {
      break;
    }
    if (!found) {
      ++index;
    }
  }

  // shouldn't use keyless for projection only
  /**
   * Currently just support shared memory usage when dealing with one keyless aggregate operation.
   * Currently just support shared memory usage for up to two target expressions.
   */
  return {keyless && found,
          index,
          init_val,
          ((num_agg_expr == 1) && (target_expr_list.size() <= 2)) ? shared_mem_support && shared_mem_valid_data_type
                                                                  : false};
}

/**
 * Supported data types for the current shared memory usage for keyless aggregates with COUNT(*)
 * Currently only for OneColKnownRange hash type.
 */
bool GroupByAndAggregate::supportedTypeForGpuSharedMemUsage(const SQLTypeInfo& target_type_info) const {
  bool result = false;
  switch (target_type_info.get_type()) {
    case SQLTypes::kTINYINT:
    case SQLTypes::kSMALLINT:
    case SQLTypes::kINT:
      result = true;
      break;
    case SQLTypes::kTEXT:
      if (target_type_info.get_compression() == EncodingType::kENCODING_DICT) {
        result = true;
      }
      break;
    default:
      break;
  }
  return result;
}

// TODO(Saman): this function is temporary and all these limitations should eventually be removed.
bool GroupByAndAggregate::supportedExprForGpuSharedMemUsage(Analyzer::Expr* expr) const {
  /*
  UNNEST operations follow a slightly different internal memory layout compared to other keyless aggregates
  Currently, we opt out of using shared memory if there is any UNNEST operation involved.
  */
  if (dynamic_cast<Analyzer::UOper*>(expr) && static_cast<Analyzer::UOper*>(expr)->get_optype() == kUNNEST) {
    return false;
  }
  return true;
}

bool GroupByAndAggregate::gpuCanHandleOrderEntries(const std::list<Analyzer::OrderEntry>& order_entries) {
  if (order_entries.size() > 1) {  // TODO(alex): lift this restriction
    return false;
  }
  for (const auto order_entry : order_entries) {
    CHECK_GE(order_entry.tle_no, 1);
    CHECK_LE(static_cast<size_t>(order_entry.tle_no), ra_exe_unit_.target_exprs.size());
    const auto target_expr = ra_exe_unit_.target_exprs[order_entry.tle_no - 1];
    if (!dynamic_cast<Analyzer::AggExpr*>(target_expr)) {
      return false;
    }
    // TODO(alex): relax the restrictions
    auto agg_expr = static_cast<Analyzer::AggExpr*>(target_expr);
    if (agg_expr->get_is_distinct() || agg_expr->get_aggtype() == kAVG || agg_expr->get_aggtype() == kMIN ||
        agg_expr->get_aggtype() == kMAX || agg_expr->get_aggtype() == kAPPROX_COUNT_DISTINCT) {
      return false;
    }
    if (agg_expr->get_arg()) {
      const auto& arg_ti = agg_expr->get_arg()->get_type_info();
      if (arg_ti.is_fp()) {
        return false;
      }
      auto expr_range_info = getExprRangeInfo(agg_expr->get_arg());
      if ((expr_range_info.hash_type_ != GroupByColRangeType::OneColKnownRange || expr_range_info.has_nulls) &&
          order_entry.is_desc == order_entry.nulls_first) {
        return false;
      }
    }
    const auto& target_ti = target_expr->get_type_info();
    CHECK(!target_ti.is_array());
    if (!target_ti.is_integer()) {
      return false;
    }
  }
  return true;
}

bool QueryMemoryDescriptor::usesGetGroupValueFast() const {
  return (hash_type == GroupByColRangeType::OneColKnownRange && !getSmallBufferSizeBytes());
}

bool QueryMemoryDescriptor::usesCachedContext() const {
  return allow_multifrag && (usesGetGroupValueFast() || hash_type == GroupByColRangeType::MultiColPerfectHash);
}

bool QueryMemoryDescriptor::threadsShareMemory() const {
  return sharing == GroupByMemSharing::Shared || sharing == GroupByMemSharing::SharedForKeylessOneColumnKnownRange;
}

bool QueryMemoryDescriptor::blocksShareMemory() const {
  if (g_cluster) {
    return true;
  }
  if (!countDescriptorsLogicallyEmpty(count_distinct_descriptors_)) {
    return true;
  }
  if (executor_->isCPUOnly() || render_output || hash_type == GroupByColRangeType::MultiCol ||
      hash_type == GroupByColRangeType::Projection || hash_type == GroupByColRangeType::MultiColPerfectHash) {
    return true;
  }
  return usesCachedContext() && !sharedMemBytes(ExecutorDeviceType::GPU) && many_entries(max_val, min_val, bucket);
}

bool QueryMemoryDescriptor::lazyInitGroups(const ExecutorDeviceType device_type) const {
  return device_type == ExecutorDeviceType::GPU && !render_output && !getSmallBufferSizeQuad() &&
         countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
}

bool QueryMemoryDescriptor::interleavedBins(const ExecutorDeviceType device_type) const {
  return interleaved_bins_on_gpu && device_type == ExecutorDeviceType::GPU;
}

size_t QueryMemoryDescriptor::sharedMemBytes(const ExecutorDeviceType device_type) const {
  CHECK(device_type == ExecutorDeviceType::CPU || device_type == ExecutorDeviceType::GPU);
  if (device_type == ExecutorDeviceType::CPU) {
    return 0;
  }
  // if performing keyless aggregate query with a single column group-by:
  if (sharing == GroupByMemSharing::SharedForKeylessOneColumnKnownRange) {
    CHECK_EQ(getRowSize(), sizeof(int64_t));  // Currently just designed for this scenario
    size_t shared_mem_size = (/*bin_count=*/entry_count + 1) * sizeof(int64_t);  // one extra for NULL values
    CHECK(shared_mem_size <= executor_->getCatalog()->get_dataMgr().cudaMgr_->maxSharedMemoryForAll);
    return shared_mem_size;
  }
  const size_t shared_mem_threshold{0};
  const size_t shared_mem_bytes{getBufferSizeBytes(ExecutorDeviceType::GPU)};
  if (!usesGetGroupValueFast() || shared_mem_bytes > shared_mem_threshold) {
    return 0;
  }
  return shared_mem_bytes;
}

bool QueryMemoryDescriptor::isWarpSyncRequired(const ExecutorDeviceType device_type) const {
  if (device_type != ExecutorDeviceType::GPU) {
    return false;
  } else {
    auto cuda_manager = executor_->getCatalog()->get_dataMgr().cudaMgr_;
    CHECK(cuda_manager);
    return cuda_manager->isArchVoltaForAll();
  }
}

bool QueryMemoryDescriptor::canOutputColumnar() const {
  return usesGetGroupValueFast() && threadsShareMemory() && blocksShareMemory() &&
         !interleavedBins(ExecutorDeviceType::GPU) && countDescriptorsLogicallyEmpty(count_distinct_descriptors_);
}

bool QueryMemoryDescriptor::sortOnGpu() const {
  return sort_on_gpu_;
}

GroupByAndAggregate::DiamondCodegen::DiamondCodegen(llvm::Value* cond,
                                                    Executor* executor,
                                                    const bool chain_to_next,
                                                    const std::string& label_prefix,
                                                    DiamondCodegen* parent,
                                                    const bool share_false_edge_with_parent)
    : executor_(executor), chain_to_next_(chain_to_next), parent_(parent) {
  if (parent_) {
    CHECK(!chain_to_next_);
  }
  cond_true_ = llvm::BasicBlock::Create(LL_CONTEXT, label_prefix + "_true", ROW_FUNC);
  if (share_false_edge_with_parent) {
    CHECK(parent);
    orig_cond_false_ = cond_false_ = parent_->cond_false_;
  } else {
    orig_cond_false_ = cond_false_ = llvm::BasicBlock::Create(LL_CONTEXT, label_prefix + "_false", ROW_FUNC);
  }

  LL_BUILDER.CreateCondBr(cond, cond_true_, cond_false_);
  LL_BUILDER.SetInsertPoint(cond_true_);
}

void GroupByAndAggregate::DiamondCodegen::setChainToNext() {
  CHECK(!parent_);
  chain_to_next_ = true;
}

void GroupByAndAggregate::DiamondCodegen::setFalseTarget(llvm::BasicBlock* cond_false) {
  CHECK(!parent_ || orig_cond_false_ != parent_->cond_false_);
  cond_false_ = cond_false;
}

GroupByAndAggregate::DiamondCodegen::~DiamondCodegen() {
  if (parent_ && orig_cond_false_ != parent_->cond_false_) {
    LL_BUILDER.CreateBr(parent_->cond_false_);
  } else if (chain_to_next_) {
    LL_BUILDER.CreateBr(cond_false_);
  }
  if (!parent_ || (!chain_to_next_ && cond_false_ != parent_->cond_false_)) {
    LL_BUILDER.SetInsertPoint(orig_cond_false_);
  }
}

void GroupByAndAggregate::patchGroupbyCall(llvm::CallInst* call_site) {
  CHECK(call_site);
  const auto func = call_site->getCalledFunction();
  const auto func_name = func->getName();
  if (func_name == "get_columnar_group_bin_offset") {
    return;
  }

  const auto arg_count = call_site->getNumArgOperands();
  const int32_t new_size_quad = static_cast<int32_t>(query_mem_desc_.getRowSize() / sizeof(int64_t));
  std::vector<llvm::Value*> args;
  size_t arg_idx{0};
  bool found{false};
  for (const auto& arg : func->args()) {
    if (arg.getName() == "row_size_quad") {
      args.push_back(LL_INT(new_size_quad));
      found = true;
    } else {
      args.push_back(call_site->getArgOperand(arg_idx));
    }
    ++arg_idx;
  }
  CHECK_EQ(true, found);
  CHECK_EQ(arg_count, arg_idx);
  llvm::ReplaceInstWithInst(call_site, llvm::CallInst::Create(func, args));
}

bool GroupByAndAggregate::codegen(llvm::Value* filter_result,
                                  llvm::Value* outerjoin_query_filter_result,
                                  llvm::BasicBlock* sc_false,
                                  const CompilationOptions& co) {
  CHECK(filter_result);

  bool can_return_error = false;
  llvm::BasicBlock* filter_false{nullptr};

  {
    const bool is_group_by = !ra_exe_unit_.groupby_exprs.empty();
    const auto query_mem_desc = getQueryMemoryDescriptor();

    if (executor_->isArchMaxwell(co.device_type_)) {
      prependForceSync();
    }
    DiamondCodegen filter_cfg(
        filter_result, executor_, !is_group_by || query_mem_desc.usesGetGroupValueFast(), "filter", nullptr, false);
    filter_false = filter_cfg.cond_false_;

    if (executor_->isOuterLoopJoin() || executor_->isOneToManyOuterHashJoin()) {
      auto match_found_ptr = executor_->cgen_state_->outer_join_match_found_;
      CHECK(match_found_ptr);
      LL_BUILDER.CreateStore(executor_->ll_bool(true), match_found_ptr);
    }

    std::unique_ptr<DiamondCodegen> nonjoin_filter_cfg;
    if (outerjoin_query_filter_result) {
      nonjoin_filter_cfg.reset(new DiamondCodegen(
          outerjoin_query_filter_result, executor_, false, "nonjoin_filter", &filter_cfg, is_group_by));
    }

    if (is_group_by) {
      if (query_mem_desc_.hash_type == GroupByColRangeType::Projection &&
          !use_streaming_top_n(ra_exe_unit_, query_mem_desc_)) {
        const auto crt_match = get_arg_by_name(ROW_FUNC, "crt_match");
        LL_BUILDER.CreateStore(LL_INT(int32_t(1)), crt_match);
        auto total_matched_ptr = get_arg_by_name(ROW_FUNC, "total_matched");
        llvm::Value* old_total_matched_val{nullptr};
        if (co.device_type_ == ExecutorDeviceType::GPU) {
          old_total_matched_val = LL_BUILDER.CreateAtomicRMW(
              llvm::AtomicRMWInst::Add, total_matched_ptr, LL_INT(int32_t(1)), llvm::AtomicOrdering::Monotonic);
        } else {
          old_total_matched_val = LL_BUILDER.CreateLoad(total_matched_ptr);
          LL_BUILDER.CreateStore(LL_BUILDER.CreateAdd(old_total_matched_val, LL_INT(int32_t(1))), total_matched_ptr);
        }
        auto old_total_matched_ptr = get_arg_by_name(ROW_FUNC, "old_total_matched");
        LL_BUILDER.CreateStore(old_total_matched_val, old_total_matched_ptr);
      }

      auto agg_out_ptr_w_idx = codegenGroupBy(co, filter_cfg);
      if (query_mem_desc.usesGetGroupValueFast() ||
          query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
        if (query_mem_desc.hash_type == GroupByColRangeType::MultiColPerfectHash) {
          filter_cfg.setChainToNext();
        }
        // Don't generate null checks if the group slot is guaranteed to be non-null,
        // as it's the case for get_group_value_fast* family.
        can_return_error = codegenAggCalls(agg_out_ptr_w_idx, {}, co);
      } else {
        {
          CHECK(!outputColumnar() || query_mem_desc.keyless_hash);
          DiamondCodegen nullcheck_cfg(LL_BUILDER.CreateICmpNE(std::get<0>(agg_out_ptr_w_idx),
                                                               llvm::ConstantPointerNull::get(llvm::PointerType::get(
                                                                   get_int_type(64, LL_CONTEXT), 0))),
                                       executor_,
                                       false,
                                       "groupby_nullcheck",
                                       &filter_cfg,
                                       false);
          codegenAggCalls(agg_out_ptr_w_idx, {}, co);
        }
        can_return_error = true;
        if (query_mem_desc_.hash_type == GroupByColRangeType::Projection &&
            use_streaming_top_n(ra_exe_unit_, query_mem_desc_)) {
          // Ignore rejection on pushing current row to top-K heap.
          LL_BUILDER.CreateRet(LL_INT(int32_t(0)));
        } else {
          LL_BUILDER.CreateRet(LL_BUILDER.CreateNeg(LL_BUILDER.CreateTrunc(
              // TODO(alex): remove the trunc once pos is converted to 32 bits
              executor_->posArg(nullptr),
              get_int_type(32, LL_CONTEXT))));
        }
      }

      if (!outputColumnar() && query_mem_desc.getRowSize() != query_mem_desc_.getRowSize()) {
        patchGroupbyCall(static_cast<llvm::CallInst*>(std::get<0>(agg_out_ptr_w_idx)));
      }

    } else {
      if (ra_exe_unit_.estimator) {
        std::stack<llvm::BasicBlock*> array_loops;
        codegenEstimator(array_loops, filter_cfg, co);
      } else {
        auto arg_it = ROW_FUNC->arg_begin();
        std::vector<llvm::Value*> agg_out_vec;
        for (int32_t i = 0; i < get_agg_count(ra_exe_unit_.target_exprs); ++i) {
          agg_out_vec.push_back(&*arg_it++);
        }
        can_return_error = codegenAggCalls(std::make_tuple(nullptr, nullptr), agg_out_vec, co);
      }
    }
  }

  if (ra_exe_unit_.inner_joins.empty()) {
    executor_->codegenInnerScanNextRowOrMatch();
  } else if (sc_false) {
    const auto saved_insert_block = LL_BUILDER.GetInsertBlock();
    LL_BUILDER.SetInsertPoint(sc_false);
    LL_BUILDER.CreateBr(filter_false);
    LL_BUILDER.SetInsertPoint(saved_insert_block);
  }

  return can_return_error;
}

llvm::Value* GroupByAndAggregate::codegenOutputSlot(llvm::Value* groups_buffer,
                                                    const CompilationOptions& co,
                                                    DiamondCodegen& diamond_codegen) {
  CHECK(query_mem_desc_.hash_type == GroupByColRangeType::Projection);
  CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
  const auto group_expr = ra_exe_unit_.groupby_exprs.front();
  CHECK(!group_expr);
  if (!outputColumnar()) {
    CHECK_EQ(size_t(0), query_mem_desc_.getRowSize() % sizeof(int64_t));
  }
  const int32_t row_size_quad = outputColumnar() ? 0 : query_mem_desc_.getRowSize() / sizeof(int64_t);
  if (use_streaming_top_n(ra_exe_unit_, query_mem_desc_)) {
    const auto& only_order_entry = ra_exe_unit_.sort_info.order_entries.front();
    CHECK_GE(only_order_entry.tle_no, int(1));
    const size_t target_idx = only_order_entry.tle_no - 1;
    CHECK_LT(target_idx, ra_exe_unit_.target_exprs.size());
    const auto order_entry_expr = ra_exe_unit_.target_exprs[target_idx];
    const auto chosen_bytes = static_cast<size_t>(query_mem_desc_.agg_col_widths[target_idx].compact);
    auto order_entry_lv =
        executor_->castToTypeIn(executor_->codegen(order_entry_expr, true, co).front(), chosen_bytes * 8);
    const uint32_t n = ra_exe_unit_.sort_info.offset + ra_exe_unit_.sort_info.limit;
    std::string fname = "get_bin_from_k_heap";
    const auto& oe_ti = order_entry_expr->get_type_info();
    llvm::Value* null_key_lv = nullptr;
    if (oe_ti.is_integer() || oe_ti.is_decimal() || oe_ti.is_time()) {
      const size_t bit_width = order_entry_lv->getType()->getIntegerBitWidth();
      switch (bit_width) {
        case 32:
          null_key_lv = LL_INT(static_cast<int32_t>(inline_int_null_val(oe_ti)));
          break;
        case 64:
          null_key_lv = LL_INT(static_cast<int64_t>(inline_int_null_val(oe_ti)));
          break;
        default:
          CHECK(false);
      }
      fname += "_int" + std::to_string(bit_width) + "_t";
    } else {
      CHECK(oe_ti.is_fp());
      if (order_entry_lv->getType()->isDoubleTy()) {
        null_key_lv = LL_FP(static_cast<double>(inline_fp_null_val(oe_ti)));
      } else {
        null_key_lv = LL_FP(static_cast<float>(inline_fp_null_val(oe_ti)));
      }
      fname += order_entry_lv->getType()->isDoubleTy() ? "_double" : "_float";
    }
    return emitCall(fname,
                    {groups_buffer,
                     LL_INT(n),
                     LL_INT(row_size_quad),
                     LL_INT(static_cast<uint32_t>(query_mem_desc_.getColOffInBytes(0, target_idx))),
                     LL_BOOL(only_order_entry.is_desc),
                     LL_BOOL(!order_entry_expr->get_type_info().get_notnull()),
                     LL_BOOL(only_order_entry.nulls_first),
                     null_key_lv,
                     order_entry_lv});
  } else {
    const auto group_expr_lv = LL_BUILDER.CreateLoad(get_arg_by_name(ROW_FUNC, "old_total_matched"));
    return emitCall("get_scan_output_slot",
                    {groups_buffer,
                     LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count)),
                     group_expr_lv,
                     executor_->posArg(nullptr),
                     LL_INT(row_size_quad)});
  }
}

std::tuple<llvm::Value*, llvm::Value*> GroupByAndAggregate::codegenGroupBy(const CompilationOptions& co,
                                                                           DiamondCodegen& diamond_codegen) {
  auto arg_it = ROW_FUNC->arg_begin();
  auto groups_buffer = arg_it++;

  const int32_t row_size_quad = outputColumnar() ? 0 : query_mem_desc_.getRowSize() / sizeof(int64_t);

  std::stack<llvm::BasicBlock*> array_loops;

  if (query_mem_desc_.hash_type == GroupByColRangeType::Projection) {
    return std::tuple<llvm::Value*, llvm::Value*>{codegenOutputSlot(&*groups_buffer, co, diamond_codegen), nullptr};
  }

  switch (query_mem_desc_.hash_type) {
    case GroupByColRangeType::OneColKnownRange:
    case GroupByColRangeType::OneColGuessedRange:
    case GroupByColRangeType::Scan: {
      CHECK_EQ(size_t(1), ra_exe_unit_.groupby_exprs.size());
      const auto group_expr = ra_exe_unit_.groupby_exprs.front();
      const auto group_expr_lvs = executor_->groupByColumnCodegen(
          group_expr.get(),
          sizeof(int64_t),
          co,
          query_mem_desc_.has_nulls,
          query_mem_desc_.max_val + (query_mem_desc_.bucket ? query_mem_desc_.bucket : 1),
          diamond_codegen,
          array_loops,
          query_mem_desc_.threadsShareMemory());
      const auto group_expr_lv = group_expr_lvs.translated_value;
      auto small_groups_buffer = arg_it;
      if (query_mem_desc_.usesGetGroupValueFast()) {
        std::string get_group_fn_name{outputColumnar() && !query_mem_desc_.keyless_hash
                                          ? "get_columnar_group_bin_offset"
                                          : "get_group_value_fast"};
        if (query_mem_desc_.keyless_hash) {
          get_group_fn_name += "_keyless";
        }
        if (query_mem_desc_.interleavedBins(co.device_type_)) {
          CHECK(!outputColumnar());
          CHECK(query_mem_desc_.keyless_hash);
          get_group_fn_name += "_semiprivate";
        }
        std::vector<llvm::Value*> get_group_fn_args{&*groups_buffer, &*group_expr_lv};
        if (group_expr_lvs.original_value && get_group_fn_name == "get_group_value_fast" &&
            query_mem_desc_.must_use_baseline_sort) {
          get_group_fn_name += "_with_original_key";
          get_group_fn_args.push_back(group_expr_lvs.original_value);
        }
        get_group_fn_args.push_back(LL_INT(query_mem_desc_.min_val));
        get_group_fn_args.push_back(LL_INT(query_mem_desc_.bucket));
        if (!query_mem_desc_.keyless_hash) {
          if (!outputColumnar()) {
            get_group_fn_args.push_back(LL_INT(row_size_quad));
          }
        } else {
          CHECK(!outputColumnar());
          get_group_fn_args.push_back(LL_INT(row_size_quad));
          if (query_mem_desc_.interleavedBins(co.device_type_)) {
            auto warp_idx = emitCall("thread_warp_idx", {LL_INT(executor_->warpSize())});
            get_group_fn_args.push_back(warp_idx);
            get_group_fn_args.push_back(LL_INT(executor_->warpSize()));
          }
        }
        if (get_group_fn_name == "get_columnar_group_bin_offset") {
          return std::make_tuple(&*groups_buffer, emitCall(get_group_fn_name, get_group_fn_args));
        }
        return std::make_tuple(emitCall(get_group_fn_name, get_group_fn_args), nullptr);
      } else {
        ++arg_it;
        ++arg_it;
        ++arg_it;
        if (group_expr) {
          LOG(INFO) << "Use get_group_value_one_key";
        }
        return std::make_tuple(
            emitCall(co.with_dynamic_watchdog_ ? "get_group_value_one_key_with_watchdog" : "get_group_value_one_key",
                     {&*groups_buffer,
                      LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count)),
                      &*small_groups_buffer,
                      LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count_small)),
                      &*group_expr_lv,
                      LL_INT(query_mem_desc_.min_val),
                      LL_INT(row_size_quad),
                      &*(++arg_it)}),
            nullptr);
      }
      break;
    }
    case GroupByColRangeType::MultiCol:
    case GroupByColRangeType::MultiColPerfectHash: {
      auto key_size_lv = LL_INT(static_cast<int32_t>(query_mem_desc_.group_col_widths.size()));
// create the key buffer
#ifdef ENABLE_KEY_COMPACTION
      const auto key_width = query_mem_desc_.getEffectiveKeyWidth();
      llvm::Value* group_key =
          query_mem_desc_.hash_type == GroupByColRangeType::MultiCol && key_width == sizeof(int32_t)
              ? LL_BUILDER.CreateAlloca(llvm::Type::getInt32Ty(LL_CONTEXT), key_size_lv)
              : LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
#else
      const auto key_width = sizeof(int64_t);
      llvm::Value* group_key = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), key_size_lv);
#endif
      int32_t subkey_idx = 0;
      for (const auto group_expr : ra_exe_unit_.groupby_exprs) {
        auto col_range_info = getExprRangeInfo(group_expr.get());
        const auto group_expr_lvs = executor_->groupByColumnCodegen(
            group_expr.get(),
            key_width,
            co,
            col_range_info.has_nulls && query_mem_desc_.hash_type != GroupByColRangeType::MultiCol,
            col_range_info.max + (col_range_info.bucket ? col_range_info.bucket : 1),
            diamond_codegen,
            array_loops,
            query_mem_desc_.threadsShareMemory());
        const auto group_expr_lv = group_expr_lvs.translated_value;
        // store the sub-key to the buffer
        LL_BUILDER.CreateStore(group_expr_lv, LL_BUILDER.CreateGEP(group_key, LL_INT(subkey_idx++)));
      }
      ++arg_it;
      ++arg_it;
      ++arg_it;
      auto perfect_hash_func = query_mem_desc_.hash_type == GroupByColRangeType::MultiColPerfectHash
                                   ? codegenPerfectHashFunction()
                                   : nullptr;
      if (perfect_hash_func) {
        auto hash_lv = LL_BUILDER.CreateCall(perfect_hash_func, std::vector<llvm::Value*>{group_key});
        return std::make_tuple(emitCall("get_matching_group_value_perfect_hash",
                                        {&*groups_buffer, hash_lv, group_key, key_size_lv, LL_INT(row_size_quad)}),
                               nullptr);
      }
#ifdef ENABLE_KEY_COMPACTION
      if (group_key->getType() != llvm::Type::getInt64PtrTy(LL_CONTEXT)) {
        CHECK(query_mem_desc_.hash_type == GroupByColRangeType::MultiCol && key_width == sizeof(int32_t));
        group_key = LL_BUILDER.CreatePointerCast(group_key, llvm::Type::getInt64PtrTy(LL_CONTEXT));
      }
#endif
      return std::make_tuple(emitCall(co.with_dynamic_watchdog_ ? "get_group_value_with_watchdog" : "get_group_value",
                                      {&*groups_buffer,
                                       LL_INT(static_cast<int32_t>(query_mem_desc_.entry_count)),
                                       &*group_key,
                                       &*key_size_lv,
                                       LL_INT(static_cast<int32_t>(key_width)),
                                       LL_INT(row_size_quad),
                                       &*(++arg_it)}),
                             nullptr);
      break;
    }
    default:
      CHECK(false);
      break;
  }

  CHECK(false);
  return std::make_tuple(nullptr, nullptr);
}

llvm::Function* GroupByAndAggregate::codegenPerfectHashFunction() {
  CHECK_GT(ra_exe_unit_.groupby_exprs.size(), size_t(1));
  auto ft = llvm::FunctionType::get(get_int_type(32, LL_CONTEXT),
                                    std::vector<llvm::Type*>{llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)},
                                    false);
  auto key_hash_func =
      llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "perfect_key_hash", executor_->cgen_state_->module_);
  executor_->cgen_state_->helper_functions_.push_back(key_hash_func);
  mark_function_always_inline(key_hash_func);
  auto& key_buff_arg = *key_hash_func->args().begin();
  llvm::Value* key_buff_lv = &key_buff_arg;
  auto bb = llvm::BasicBlock::Create(LL_CONTEXT, "entry", key_hash_func);
  llvm::IRBuilder<> key_hash_func_builder(bb);
  llvm::Value* hash_lv{llvm::ConstantInt::get(get_int_type(64, LL_CONTEXT), 0)};
  std::vector<int64_t> cardinalities;
  for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto col_range_info = getExprRangeInfo(groupby_expr.get());
    CHECK(col_range_info.hash_type_ == GroupByColRangeType::OneColKnownRange);
    cardinalities.push_back(getBucketedCardinality(col_range_info));
  }
  size_t dim_idx = 0;
  for (const auto groupby_expr : ra_exe_unit_.groupby_exprs) {
    auto key_comp_lv = key_hash_func_builder.CreateLoad(key_hash_func_builder.CreateGEP(key_buff_lv, LL_INT(dim_idx)));
    auto col_range_info = getExprRangeInfo(groupby_expr.get());
    auto crt_term_lv = key_hash_func_builder.CreateSub(key_comp_lv, LL_INT(col_range_info.min));
    if (col_range_info.bucket) {
      crt_term_lv = key_hash_func_builder.CreateSDiv(crt_term_lv, LL_INT(col_range_info.bucket));
    }
    for (size_t prev_dim_idx = 0; prev_dim_idx < dim_idx; ++prev_dim_idx) {
      crt_term_lv = key_hash_func_builder.CreateMul(crt_term_lv, LL_INT(cardinalities[prev_dim_idx]));
    }
    hash_lv = key_hash_func_builder.CreateAdd(hash_lv, crt_term_lv);
    ++dim_idx;
  }
  key_hash_func_builder.CreateRet(key_hash_func_builder.CreateTrunc(hash_lv, get_int_type(32, LL_CONTEXT)));
  return key_hash_func;
}

namespace {

std::vector<std::string> agg_fn_base_names(const TargetInfo& target_info) {
  const auto& chosen_type = get_compact_type(target_info);
  if (!target_info.is_agg || target_info.agg_kind == kLAST_SAMPLE) {
    if (chosen_type.is_geometry()) {
      return std::vector<std::string>(2 * chosen_type.get_physical_coord_cols(), "agg_id");
    }
    if (chosen_type.is_varlen()) {
      return {"agg_id", "agg_id"};
    }
    return {"agg_id"};
  }
  switch (target_info.agg_kind) {
    case kAVG:
      return {"agg_sum", "agg_count"};
    case kCOUNT:
      return {target_info.is_distinct ? "agg_count_distinct" : "agg_count"};
    case kMAX:
      return {"agg_max"};
    case kMIN:
      return {"agg_min"};
    case kSUM:
      return {"agg_sum"};
    case kAPPROX_COUNT_DISTINCT:
      return {"agg_approximate_count_distinct"};
    case kLAST_SAMPLE:
      return {"agg_id"};
    default:
      abort();
  }
}

}  // namespace

llvm::Value* GroupByAndAggregate::convertNullIfAny(const SQLTypeInfo& arg_type,
                                                   const SQLTypeInfo& agg_type,
                                                   const size_t chosen_bytes,
                                                   llvm::Value* target) {
  bool need_conversion{false};
  llvm::Value* arg_null{nullptr};
  llvm::Value* agg_null{nullptr};
  llvm::Value* target_to_cast{target};
  if (arg_type.is_fp()) {
    arg_null = executor_->inlineFpNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->inlineFpNull(agg_type);
      if (!static_cast<llvm::ConstantFP*>(arg_null)->isExactlyValue(
              static_cast<llvm::ConstantFP*>(agg_null)->getValueAPF())) {
        need_conversion = true;
      }
    } else {
      // TODO(miyu): invalid case for now
      CHECK(false);
    }
  } else {
    arg_null = executor_->inlineIntNull(arg_type);
    if (agg_type.is_fp()) {
      agg_null = executor_->inlineFpNull(agg_type);
      need_conversion = true;
      target_to_cast = executor_->castToFP(target);
    } else {
      agg_null = executor_->inlineIntNull(agg_type);
      if ((static_cast<llvm::ConstantInt*>(arg_null)->getBitWidth() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getBitWidth()) ||
          (static_cast<llvm::ConstantInt*>(arg_null)->getValue() !=
           static_cast<llvm::ConstantInt*>(agg_null)->getValue())) {
        need_conversion = true;
      }
    }
  }
  if (need_conversion) {
    auto cmp =
        arg_type.is_fp() ? LL_BUILDER.CreateFCmpOEQ(target, arg_null) : LL_BUILDER.CreateICmpEQ(target, arg_null);
    return LL_BUILDER.CreateSelect(cmp, agg_null, executor_->castToTypeIn(target_to_cast, chosen_bytes << 3));
  } else {
    return target;
  }
}

#ifdef ENABLE_COMPACTION
bool GroupByAndAggregate::detectOverflowAndUnderflow(llvm::Value* agg_col_val,
                                                     llvm::Value* val,
                                                     const TargetInfo& agg_info,
                                                     const size_t chosen_bytes,
                                                     const bool need_skip_null,
                                                     const std::string& agg_base_name) {
  const auto& chosen_type = get_compact_type(agg_info);
  if (!agg_info.is_agg || (agg_base_name != "agg_sum" && agg_base_name != "agg_count") ||
      is_distinct_target(agg_info) || !chosen_type.is_integer()) {
    return false;
  }
  auto bb_no_null = LL_BUILDER.GetInsertBlock();
  auto bb_pass = llvm::BasicBlock::Create(LL_CONTEXT, ".bb_pass", ROW_FUNC, 0);
  bb_pass->moveAfter(bb_no_null);
  if (need_skip_null) {
    auto agg_null = executor_->castToTypeIn(executor_->inlineIntNull(chosen_type), (chosen_bytes << 3));
    auto null_check = LL_BUILDER.CreateICmpEQ(agg_col_val, agg_null);
    if (agg_base_name != "agg_count") {
      null_check = LL_BUILDER.CreateOr(null_check, LL_BUILDER.CreateICmpEQ(val, agg_null));
    }

    bb_no_null = llvm::BasicBlock::Create(LL_CONTEXT, ".no_null", ROW_FUNC, bb_pass);
    LL_BUILDER.CreateCondBr(null_check, bb_pass, bb_no_null);
    LL_BUILDER.SetInsertPoint(bb_no_null);
  }

  llvm::Value* chosen_max{nullptr};
  llvm::Value* chosen_min{nullptr};
  std::tie(chosen_max, chosen_min) = executor_->inlineIntMaxMin(chosen_bytes, agg_base_name != "agg_count");

  llvm::Value* detected{nullptr};
  if (agg_base_name == "agg_count") {
    auto const_one = llvm::ConstantInt::get(get_int_type(chosen_bytes << 3, LL_CONTEXT), 1);
    detected = LL_BUILDER.CreateICmpUGT(agg_col_val, LL_BUILDER.CreateSub(chosen_max, const_one));
  } else {
    auto const_zero = llvm::ConstantInt::get(get_int_type(chosen_bytes << 3, LL_CONTEXT), 0);
    auto overflow = LL_BUILDER.CreateAnd(LL_BUILDER.CreateICmpSGT(val, const_zero),
                                         LL_BUILDER.CreateICmpSGT(agg_col_val, LL_BUILDER.CreateSub(chosen_max, val)));
    auto underflow = LL_BUILDER.CreateAnd(LL_BUILDER.CreateICmpSLT(val, const_zero),
                                          LL_BUILDER.CreateICmpSLT(agg_col_val, LL_BUILDER.CreateSub(chosen_min, val)));
    detected = LL_BUILDER.CreateOr(overflow, underflow);
  }
  auto bb_fail = llvm::BasicBlock::Create(LL_CONTEXT, ".bb_fail", ROW_FUNC, bb_pass);
  LL_BUILDER.CreateCondBr(detected, bb_fail, bb_pass, llvm::MDBuilder(LL_CONTEXT).createBranchWeights(1, 100));
  LL_BUILDER.SetInsertPoint(bb_fail);
  LL_BUILDER.CreateRet(LL_INT(Executor::ERR_OVERFLOW_OR_UNDERFLOW));

  LL_BUILDER.SetInsertPoint(bb_pass);
  return true;
}
#endif  // ENABLE_COMPACTION

bool GroupByAndAggregate::codegenAggCalls(const std::tuple<llvm::Value*, llvm::Value*>& agg_out_ptr_w_idx,
                                          const std::vector<llvm::Value*>& agg_out_vec,
                                          const CompilationOptions& co) {
  // TODO(alex): unify the two cases, the output for non-group by queries
  //             should be a contiguous buffer
  const bool is_group_by{std::get<0>(agg_out_ptr_w_idx)};
  bool can_return_error = false;
  if (is_group_by) {
    CHECK(agg_out_vec.empty());
  } else {
    CHECK(!agg_out_vec.empty());
  }
  int32_t agg_out_off{0};
  for (size_t target_idx = 0; target_idx < ra_exe_unit_.target_exprs.size(); ++target_idx) {
    auto target_expr = ra_exe_unit_.target_exprs[target_idx];
    CHECK(target_expr);
    if (query_mem_desc_.agg_col_widths[agg_out_off].compact == 0) {
      CHECK(!dynamic_cast<const Analyzer::AggExpr*>(target_expr));
      ++agg_out_off;
      continue;
    }
    if (dynamic_cast<Analyzer::UOper*>(target_expr) &&
        static_cast<Analyzer::UOper*>(target_expr)->get_optype() == kUNNEST) {
      throw std::runtime_error("UNNEST not supported in the projection list yet.");
    }
    auto agg_info = target_info(target_expr);
    auto arg_expr = agg_arg(target_expr);
    if (arg_expr) {
      if (agg_info.agg_kind == kLAST_SAMPLE) {
        agg_info.skip_null_val = false;
      } else if (query_mem_desc_.hash_type == GroupByColRangeType::Scan) {
        agg_info.skip_null_val = true;
      } else if (constrained_not_null(arg_expr, ra_exe_unit_.quals)) {
        agg_info.skip_null_val = false;
      }
    }
    const auto agg_fn_names = agg_fn_base_names(agg_info);
    auto target_lvs = codegenAggArg(target_expr, co);
    if ((executor_->plan_state_->isLazyFetchColumn(target_expr) || !is_group_by) &&
        static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_out_off].compact) < sizeof(int64_t)) {
      // TODO(miyu): enable different byte width in the layout w/o padding
      throw CompilationRetryNoCompaction();
    }
    llvm::Value* str_target_lv{nullptr};
    const bool agg_has_geo = agg_info.is_agg ? agg_info.agg_arg_type.is_geometry() : agg_info.sql_type.is_geometry();
    if (target_lvs.size() == 3 && !agg_has_geo) {
      // none encoding string, pop the packed pointer + length since
      // it's only useful for IS NULL checks and assumed to be only
      // two components (pointer and length) for the purpose of projection
      str_target_lv = target_lvs.front();
      target_lvs.erase(target_lvs.begin());
    }
    if (agg_info.sql_type.is_geometry()) {
      // Geo cols are expanded to the physical coord cols. Each physical coord col is an array. Ensure that the target
      // values generated match the number of agg functions before continuing
      if (target_lvs.size() < agg_fn_names.size()) {
        CHECK_EQ(target_lvs.size(), agg_fn_names.size() / 2);
        std::vector<llvm::Value*> new_target_lvs;
        new_target_lvs.reserve(agg_fn_names.size());
        for (const auto& target_lv : target_lvs) {
          new_target_lvs.push_back(target_lv);
          new_target_lvs.push_back(target_lv);
        }
        target_lvs = new_target_lvs;
      }
    }
    if (target_lvs.size() < agg_fn_names.size()) {
      CHECK_EQ(size_t(1), target_lvs.size());
      CHECK_EQ(size_t(2), agg_fn_names.size());
      for (size_t i = 1; i < agg_fn_names.size(); ++i) {
        target_lvs.push_back(target_lvs.front());
      }
    } else {
      if (agg_has_geo) {
        if (agg_info.is_agg) {
          CHECK_EQ(static_cast<size_t>(agg_info.agg_arg_type.get_physical_coord_cols()), target_lvs.size());
        } else {
          CHECK_EQ(static_cast<size_t>(2 * agg_info.sql_type.get_physical_coord_cols()), target_lvs.size());
          CHECK_EQ(agg_fn_names.size(), target_lvs.size());
        }
      } else {
        CHECK(str_target_lv || (agg_fn_names.size() == target_lvs.size()));
        CHECK(target_lvs.size() == 1 || target_lvs.size() == 2);
      }
    }
    uint32_t col_off{0};
    const bool is_simple_count = agg_info.is_agg && agg_info.agg_kind == kCOUNT && !agg_info.is_distinct;
    if (co.device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory() && is_simple_count &&
        (!arg_expr || arg_expr->get_type_info().get_notnull())) {
      CHECK_EQ(size_t(1), agg_fn_names.size());
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[agg_out_off].compact;
      llvm::Value* agg_col_ptr{nullptr};
      if (is_group_by) {
        if (outputColumnar()) {
          col_off = query_mem_desc_.getColOffInBytes(0, agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          CHECK(std::get<1>(agg_out_ptr_w_idx));
          auto offset = LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              offset);
        } else {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              LL_INT(col_off));
        }
      }

      if (chosen_bytes != sizeof(int32_t)) {
        CHECK_EQ(8, chosen_bytes);
        if (g_bigint_count) {
          const auto acc_i64 = LL_BUILDER.CreateBitCast(is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off],
                                                        llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0));
          LL_BUILDER.CreateAtomicRMW(
              llvm::AtomicRMWInst::Add, acc_i64, LL_INT(int64_t(1)), llvm::AtomicOrdering::Monotonic);
        } else {
          const auto acc_i32 = LL_BUILDER.CreateBitCast(is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off],
                                                        llvm::PointerType::get(get_int_type(32, LL_CONTEXT), 0));
          LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add, acc_i32, LL_INT(1), llvm::AtomicOrdering::Monotonic);
        }
      } else {
        const auto acc_i32 = (is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off]);
        if (query_mem_desc_.sharing == GroupByMemSharing::SharedForKeylessOneColumnKnownRange) {
          // Atomic operation on address space level 3 (Shared):
          const auto shared_acc_i32 = LL_BUILDER.CreatePointerCast(acc_i32, llvm::Type::getInt32PtrTy(LL_CONTEXT, 3));
          LL_BUILDER.CreateAtomicRMW(
              llvm::AtomicRMWInst::Add, shared_acc_i32, LL_INT(1), llvm::AtomicOrdering::Monotonic);
        } else {
          LL_BUILDER.CreateAtomicRMW(llvm::AtomicRMWInst::Add, acc_i32, LL_INT(1), llvm::AtomicOrdering::Monotonic);
        }
      }
      ++agg_out_off;
      continue;
    }
    size_t target_lv_idx = 0;
    const bool lazy_fetched{executor_->plan_state_->isLazyFetchColumn(target_expr)};
    for (const auto& agg_base_name : agg_fn_names) {
      if (agg_info.is_distinct && arg_expr->get_type_info().is_array()) {
        CHECK_EQ(static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_out_off].actual), sizeof(int64_t));
        // TODO(miyu): check if buffer may be columnar here
        CHECK(!outputColumnar());
        const auto& elem_ti = arg_expr->get_type_info().get_elem_type();
        if (is_group_by) {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % sizeof(int64_t));
          col_off /= sizeof(int64_t);
        }
        executor_->cgen_state_->emitExternalCall(
            "agg_count_distinct_array_" + numeric_type_name(elem_ti),
            llvm::Type::getVoidTy(LL_CONTEXT),
            {is_group_by ? LL_BUILDER.CreateGEP(std::get<0>(agg_out_ptr_w_idx), LL_INT(col_off))
                         : agg_out_vec[agg_out_off],
             target_lvs[target_lv_idx],
             executor_->posArg(arg_expr),
             elem_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(elem_ti))
                             : static_cast<llvm::Value*>(executor_->inlineIntNull(elem_ti))});
        ++agg_out_off;
        ++target_lv_idx;
        continue;
      }

      llvm::Value* agg_col_ptr{nullptr};
      const auto chosen_bytes = static_cast<size_t>(query_mem_desc_.agg_col_widths[agg_out_off].compact);
      const auto& chosen_type = get_compact_type(agg_info);
      const auto& arg_type = ((arg_expr && arg_expr->get_type_info().get_type() != kNULLT) && !agg_info.is_distinct)
                                 ? agg_info.agg_arg_type
                                 : agg_info.sql_type;
      const bool is_fp_arg = !lazy_fetched && arg_type.get_type() != kNULLT && arg_type.is_fp();
      if (is_group_by) {
        if (outputColumnar()) {
          col_off = query_mem_desc_.getColOffInBytes(0, agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          CHECK(std::get<1>(agg_out_ptr_w_idx));
          auto offset = LL_BUILDER.CreateAdd(std::get<1>(agg_out_ptr_w_idx), LL_INT(col_off));
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              offset);
        } else {
          col_off = query_mem_desc_.getColOnlyOffInBytes(agg_out_off);
          CHECK_EQ(size_t(0), col_off % chosen_bytes);
          col_off /= chosen_bytes;
          agg_col_ptr = LL_BUILDER.CreateGEP(
              LL_BUILDER.CreateBitCast(std::get<0>(agg_out_ptr_w_idx),
                                       llvm::PointerType::get(get_int_type((chosen_bytes << 3), LL_CONTEXT), 0)),
              LL_INT(col_off));
        }
      }

      const bool float_argument_input = takes_float_argument(agg_info);
      const bool is_count_in_avg = agg_info.agg_kind == kAVG && target_lv_idx == 1;
      // The count component of an average should never be compacted.
      const auto agg_chosen_bytes = float_argument_input && !is_count_in_avg ? sizeof(float) : chosen_bytes;
      if (float_argument_input) {
        CHECK_GE(chosen_bytes, sizeof(float));
      }

      auto target_lv = target_lvs[target_lv_idx];
      const auto needs_unnest_double_patch = needsUnnestDoublePatch(target_lv, agg_base_name, co);
      const auto need_skip_null = !needs_unnest_double_patch && agg_info.skip_null_val;
      if (!needs_unnest_double_patch) {
        if (need_skip_null && agg_info.agg_kind != kCOUNT) {
          target_lv = convertNullIfAny(arg_expr->get_type_info(), arg_type, agg_chosen_bytes, target_lv);
        } else if (is_fp_arg) {
          target_lv = executor_->castToFP(target_lv);
        }
        if (!dynamic_cast<const Analyzer::AggExpr*>(target_expr) || arg_expr) {
          target_lv = executor_->castToTypeIn(target_lv, (agg_chosen_bytes << 3));
        }
      }

      std::vector<llvm::Value*> agg_args{
          executor_->castToIntPtrTyIn((is_group_by ? agg_col_ptr : agg_out_vec[agg_out_off]), (agg_chosen_bytes << 3)),
          (is_simple_count && !arg_expr)
              ? (agg_chosen_bytes == sizeof(int32_t) ? LL_INT(int32_t(0)) : LL_INT(int64_t(0)))
              : (is_simple_count && arg_expr && str_target_lv ? str_target_lv : target_lv)};
      std::string agg_fname{agg_base_name};
      if (is_fp_arg) {
        if (!lazy_fetched) {
          if (agg_chosen_bytes == sizeof(float)) {
            CHECK_EQ(arg_type.get_type(), kFLOAT);
            agg_fname += "_float";
          } else {
            CHECK_EQ(agg_chosen_bytes, sizeof(double));
            agg_fname += "_double";
          }
        }
      } else if (agg_chosen_bytes == sizeof(int32_t)) {
        agg_fname += "_int32";
      }

      if (is_distinct_target(agg_info)) {
        CHECK_EQ(agg_chosen_bytes, sizeof(int64_t));
        CHECK(!chosen_type.is_fp());
        codegenCountDistinct(target_idx, target_expr, agg_args, query_mem_desc_, co.device_type_);
      } else {
        const auto& arg_ti = agg_info.agg_arg_type;
        if (need_skip_null && !arg_ti.is_geometry()) {
          agg_fname += "_skip_val";
          auto null_lv =
              executor_->castToTypeIn(arg_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(arg_ti))
                                                     : static_cast<llvm::Value*>(executor_->inlineIntNull(arg_ti)),
                                      (agg_chosen_bytes << 3));
          agg_args.push_back(null_lv);
        }
        if (!agg_info.is_distinct) {
          if (co.device_type_ == ExecutorDeviceType::GPU && query_mem_desc_.threadsShareMemory()) {
            agg_fname += "_shared";
            if (needs_unnest_double_patch) {
              agg_fname = patch_agg_fname(agg_fname);
            }
          }

          auto old_val = emitCall(agg_fname, agg_args);

#ifdef ENABLE_COMPACTION
          CHECK_LE(size_t(2), agg_args.size());
          can_return_error = detectOverflowAndUnderflow(
              old_val, agg_args[1], agg_info, agg_chosen_bytes, need_skip_null, agg_base_name);
#else
          static_cast<void>(old_val);
#endif
        }
      }
      ++agg_out_off;
      ++target_lv_idx;
    }
  }
  for (auto target_expr : ra_exe_unit_.target_exprs) {
    CHECK(target_expr);
    executor_->plan_state_->isLazyFetchColumn(target_expr);
  }

  return can_return_error;
}

void GroupByAndAggregate::codegenEstimator(std::stack<llvm::BasicBlock*>& array_loops,
                                           GroupByAndAggregate::DiamondCodegen& diamond_codegen,
                                           const CompilationOptions& co) {
  const auto& estimator_arg = ra_exe_unit_.estimator->getArgument();
  auto estimator_comp_count_lv = LL_INT(static_cast<int32_t>(estimator_arg.size()));
  auto estimator_key_lv = LL_BUILDER.CreateAlloca(llvm::Type::getInt64Ty(LL_CONTEXT), estimator_comp_count_lv);
  int32_t subkey_idx = 0;
  for (const auto estimator_arg_comp : estimator_arg) {
    const auto estimator_arg_comp_lvs = executor_->groupByColumnCodegen(estimator_arg_comp.get(),
                                                                        query_mem_desc_.getEffectiveKeyWidth(),
                                                                        co,
                                                                        false,
                                                                        0,
                                                                        diamond_codegen,
                                                                        array_loops,
                                                                        true);
    CHECK(!estimator_arg_comp_lvs.original_value);
    const auto estimator_arg_comp_lv = estimator_arg_comp_lvs.translated_value;
    // store the sub-key to the buffer
    LL_BUILDER.CreateStore(estimator_arg_comp_lv, LL_BUILDER.CreateGEP(estimator_key_lv, LL_INT(subkey_idx++)));
  }
  const auto int8_ptr_ty = llvm::PointerType::get(get_int_type(8, LL_CONTEXT), 0);
  const auto bitmap = LL_BUILDER.CreateBitCast(&*ROW_FUNC->arg_begin(), int8_ptr_ty);
  const auto key_bytes = LL_BUILDER.CreateBitCast(estimator_key_lv, int8_ptr_ty);
  const auto estimator_comp_bytes_lv = LL_INT(static_cast<int32_t>(estimator_arg.size() * sizeof(int64_t)));
  const auto bitmap_size_lv = LL_INT(static_cast<uint32_t>(ra_exe_unit_.estimator->getEstimatorBufferSize()));
  emitCall("linear_probabilistic_count", {bitmap, &*bitmap_size_lv, key_bytes, &*estimator_comp_bytes_lv});
}

void GroupByAndAggregate::codegenCountDistinct(const size_t target_idx,
                                               const Analyzer::Expr* target_expr,
                                               std::vector<llvm::Value*>& agg_args,
                                               const QueryMemoryDescriptor& query_mem_desc,
                                               const ExecutorDeviceType device_type) {
  const auto agg_info = target_info(target_expr);
  const auto& arg_ti = static_cast<const Analyzer::AggExpr*>(target_expr)->get_arg()->get_type_info();
  if (arg_ti.is_fp()) {
    agg_args.back() = executor_->cgen_state_->ir_builder_.CreateBitCast(
        agg_args.back(), get_int_type(64, executor_->cgen_state_->context_));
  }
  CHECK_LT(target_idx, query_mem_desc_.count_distinct_descriptors_.size());
  const auto& count_distinct_descriptor = query_mem_desc.count_distinct_descriptors_[target_idx];
  CHECK(count_distinct_descriptor.impl_type_ != CountDistinctImplType::Invalid);
  if (agg_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_args.push_back(LL_INT(int32_t(count_distinct_descriptor.bitmap_sz_bits)));
    if (device_type == ExecutorDeviceType::GPU) {
      const auto base_dev_addr = getAdditionalLiteral(-1);
      const auto base_host_addr = getAdditionalLiteral(-2);
      agg_args.push_back(base_dev_addr);
      agg_args.push_back(base_host_addr);
      emitCall("agg_approximate_count_distinct_gpu", agg_args);
    } else {
      emitCall("agg_approximate_count_distinct", agg_args);
    }
    return;
  }
  std::string agg_fname{"agg_count_distinct"};
  if (count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap) {
    agg_fname += "_bitmap";
    agg_args.push_back(LL_INT(static_cast<int64_t>(count_distinct_descriptor.min_val)));
  }
  if (agg_info.skip_null_val) {
    auto null_lv =
        executor_->castToTypeIn((arg_ti.is_fp() ? static_cast<llvm::Value*>(executor_->inlineFpNull(arg_ti))
                                                : static_cast<llvm::Value*>(executor_->inlineIntNull(arg_ti))),
                                64);
    null_lv =
        executor_->cgen_state_->ir_builder_.CreateBitCast(null_lv, get_int_type(64, executor_->cgen_state_->context_));
    agg_fname += "_skip_val";
    agg_args.push_back(null_lv);
  }
  if (device_type == ExecutorDeviceType::GPU) {
    CHECK(count_distinct_descriptor.impl_type_ == CountDistinctImplType::Bitmap);
    agg_fname += "_gpu";
    const auto base_dev_addr = getAdditionalLiteral(-1);
    const auto base_host_addr = getAdditionalLiteral(-2);
    agg_args.push_back(base_dev_addr);
    agg_args.push_back(base_host_addr);
    agg_args.push_back(LL_INT(int64_t(count_distinct_descriptor.sub_bitmap_count)));
    CHECK_EQ(size_t(0), count_distinct_descriptor.bitmapPaddedSizeBytes() % count_distinct_descriptor.sub_bitmap_count);
    agg_args.push_back(LL_INT(
        int64_t(count_distinct_descriptor.bitmapPaddedSizeBytes() / count_distinct_descriptor.sub_bitmap_count)));
  }
  emitCall(agg_fname, agg_args);
}

llvm::Value* GroupByAndAggregate::getAdditionalLiteral(const int32_t off) {
  CHECK_LT(off, 0);
  const auto lit_buff_lv = get_arg_by_name(ROW_FUNC, "literals");
  return LL_BUILDER.CreateLoad(LL_BUILDER.CreateGEP(
      LL_BUILDER.CreateBitCast(lit_buff_lv, llvm::PointerType::get(get_int_type(64, LL_CONTEXT), 0)), LL_INT(off)));
}

std::vector<llvm::Value*> GroupByAndAggregate::codegenAggArg(const Analyzer::Expr* target_expr,
                                                             const CompilationOptions& co) {
  const auto agg_expr = dynamic_cast<const Analyzer::AggExpr*>(target_expr);
  // TODO(alex): handle arrays uniformly?
  if (target_expr) {
    const auto& target_ti = target_expr->get_type_info();
    if (target_ti.is_array() && !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      const auto target_lvs = executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
      CHECK_EQ(size_t(1), target_lvs.size());
      CHECK(!agg_expr);
      const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
      const auto i8p_ty = llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
      const auto& elem_ti = target_ti.get_elem_type();
      return {executor_->cgen_state_->emitExternalCall(
                  "array_buff", i8p_ty, {target_lvs.front(), executor_->posArg(target_expr)}),
              executor_->cgen_state_->emitExternalCall("array_size",
                                                       i32_ty,
                                                       {target_lvs.front(),
                                                        executor_->posArg(target_expr),
                                                        executor_->ll_int(log2_bytes(elem_ti.get_logical_size()))})};
    }
    if (target_ti.is_geometry() && !executor_->plan_state_->isLazyFetchColumn(target_expr)) {
      const auto target_lvs = executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
      CHECK_EQ(target_ti.get_physical_coord_cols(), target_lvs.size());
      CHECK(!agg_expr);
      const auto i32_ty = get_int_type(32, executor_->cgen_state_->context_);
      const auto i8p_ty = llvm::PointerType::get(get_int_type(8, executor_->cgen_state_->context_), 0);
      std::vector<llvm::Value*> coords;
      size_t ctr = 0;
      for (const auto& target_lv : target_lvs) {
        // TODO(adb): consider adding a utility to sqltypes so we can get the types of the physical coords cols based on
        // the sqltype (e.g. TINYINT for col 0, INT for col 1 for pols / mpolys, etc). Hardcoding for now.
        // first array is the coords array (TINYINT). Subsequent arrays are regular INT.
        const size_t elem_sz = ctr == 0 ? 1 : 4;
        ctr++;
        coords.push_back(executor_->cgen_state_->emitExternalCall(
            "array_buff", i8p_ty, {target_lv, executor_->posArg(target_expr)}));
        coords.push_back(executor_->cgen_state_->emitExternalCall(
            "array_size", i32_ty, {target_lv, executor_->posArg(target_expr), executor_->ll_int(log2_bytes(elem_sz))}));
      }
      return coords;
    }
  }
  return agg_expr ? executor_->codegen(agg_expr->get_arg(), true, co)
                  : executor_->codegen(target_expr, !executor_->plan_state_->allow_lazy_fetch_, co);
}

llvm::Value* GroupByAndAggregate::emitCall(const std::string& fname, const std::vector<llvm::Value*>& args) {
  return executor_->cgen_state_->emitCall(fname, args);
}

#undef ROW_FUNC
#undef LL_FP
#undef LL_INT
#undef LL_BOOL
#undef LL_BUILDER
#undef LL_CONTEXT

size_t shard_count_for_top_groups(const RelAlgExecutionUnit& ra_exe_unit, const Catalog_Namespace::Catalog& catalog) {
  if (ra_exe_unit.sort_info.order_entries.size() != 1 || !ra_exe_unit.sort_info.limit) {
    return 0;
  }
  for (const auto& group_expr : ra_exe_unit.groupby_exprs) {
    const auto grouped_col_expr = dynamic_cast<const Analyzer::ColumnVar*>(group_expr.get());
    if (!grouped_col_expr) {
      continue;
    }
    if (grouped_col_expr->get_table_id() <= 0) {
      return 0;
    }
    const auto td = catalog.getMetadataForTable(grouped_col_expr->get_table_id());
    if (td->shardedColumnId == grouped_col_expr->get_column_id()) {
      return td->nShards;
    }
  }
  return 0;
}
