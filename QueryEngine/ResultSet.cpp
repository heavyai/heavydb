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

/**
 * @file    ResultSet.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"

#include "CudaAllocator.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Execute.h"
#include "GpuMemUtils.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "Shared/checked_alloc.h"
#include "Shared/likely.h"
#include "Shared/thread_count.h"
#include "SqlTypesLayout.h"

#include <algorithm>
#include <bitset>
#include <future>
#include <numeric>

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff,
                                   const bool buff_is_provided)
    : targets_(targets)
    , query_mem_desc_(query_mem_desc)
    , buff_(buff)
    , buff_is_provided_(buff_is_provided) {
  for (const auto& target_info : targets_) {
    if (target_info.agg_kind == kCOUNT ||
        target_info.agg_kind == kAPPROX_COUNT_DISTINCT) {
      target_init_vals_.push_back(0);
      continue;
    }
    if (!target_info.sql_type.get_notnull()) {
      int64_t init_val =
          null_val_bit_pattern(target_info.sql_type, takes_float_argument(target_info));
      target_init_vals_.push_back(target_info.is_agg ? init_val : 0);
    } else {
      target_init_vals_.push_back(target_info.is_agg ? 0xdeadbeef : 0);
    }
    if (target_info.agg_kind == kAVG) {
      target_init_vals_.push_back(0);
    } else if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_geometry()) {
      for (int i = 1; i < 2 * target_info.sql_type.get_physical_coord_cols(); i++) {
        target_init_vals_.push_back(0);
      }
    } else if (target_info.agg_kind == kSAMPLE && target_info.sql_type.is_varlen()) {
      target_init_vals_.push_back(0);
    }
  }
}

int8_t* ResultSetStorage::getUnderlyingBuffer() const {
  return buff_;
}

void ResultSet::keepFirstN(const size_t n) {
  CHECK_EQ(-1, cached_row_count_);
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  CHECK_EQ(-1, cached_row_count_);
  drop_first_ = n;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Executor* executor)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(-1)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , queue_time_ms_(0)
    , render_time_ms_(0)
    , executor_(executor)
    , estimator_buffer_(nullptr)
    , host_estimator_buffer_(nullptr)
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                     const std::vector<std::vector<int64_t>>& frag_offsets,
                     const std::vector<int64_t>& consistent_frag_sizes,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Executor* executor)
    : targets_(targets)
    , device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_(query_mem_desc)
    , crt_row_buff_idx_(0)
    , fetched_so_far_(0)
    , drop_first_(0)
    , keep_first_(0)
    , row_set_mem_owner_(row_set_mem_owner)
    , queue_time_ms_(0)
    , render_time_ms_(0)
    , executor_(executor)
    , lazy_fetch_info_(lazy_fetch_info)
    , col_buffers_{col_buffers}
    , frag_offsets_{frag_offsets}
    , consistent_frag_sizes_{consistent_frag_sizes}
    , estimator_buffer_(nullptr)
    , host_estimator_buffer_(nullptr)
    , data_mgr_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(const std::shared_ptr<const Analyzer::Estimator> estimator,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     Data_Namespace::DataMgr* data_mgr)
    : device_type_(device_type)
    , device_id_(device_id)
    , query_mem_desc_{}
    , crt_row_buff_idx_(0)
    , estimator_(estimator)
    , estimator_buffer_(nullptr)
    , host_estimator_buffer_(nullptr)
    , data_mgr_(data_mgr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(false)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {
  if (device_type == ExecutorDeviceType::GPU) {
    estimator_buffer_ = reinterpret_cast<int8_t*>(
        alloc_gpu_mem(data_mgr_, estimator_->getBufferSize(), device_id_, nullptr));
    data_mgr->getCudaMgr()->zeroDeviceMem(
        estimator_buffer_, estimator_->getBufferSize(), device_id_);
  } else {
    OOM_TRACE_PUSH(+": host_estimator_buffer_ " +
                   std::to_string(estimator_->getBufferSize()));
    host_estimator_buffer_ =
        static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  }
}

ResultSet::ResultSet(const std::string& explanation)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , queue_time_ms_(0)
    , render_time_ms_(0)
    , estimator_buffer_(nullptr)
    , host_estimator_buffer_(nullptr)
    , separate_varlen_storage_valid_(false)
    , explanation_(explanation)
    , just_explain_(true)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString) {}

ResultSet::ResultSet(int64_t queue_time_ms,
                     int64_t render_time_ms,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner)
    : device_type_(ExecutorDeviceType::CPU)
    , device_id_(-1)
    , fetched_so_far_(0)
    , queue_time_ms_(queue_time_ms)
    , render_time_ms_(render_time_ms)
    , estimator_buffer_(nullptr)
    , host_estimator_buffer_(nullptr)
    , separate_varlen_storage_valid_(false)
    , just_explain_(true)
    , cached_row_count_(-1)
    , geo_return_type_(GeoReturnType::WktString){};

ResultSet::~ResultSet() {
  if (storage_) {
    CHECK(storage_->getUnderlyingBuffer());
    if (!storage_->buff_is_provided_) {
      free(storage_->getUnderlyingBuffer());
    }
  }
  for (auto& storage : appended_storage_) {
    if (storage && !storage->buff_is_provided_) {
      free(storage->getUnderlyingBuffer());
    }
  }
  if (host_estimator_buffer_) {
    CHECK(device_type_ == ExecutorDeviceType::CPU || estimator_buffer_);
    free(host_estimator_buffer_);
  }
}

ExecutorDeviceType ResultSet::getDeviceType() const {
  return device_type_;
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  OOM_TRACE_PUSH(+": size " +
                 std::to_string(query_mem_desc_.getBufferSizeBytes(device_type_)));
  auto buff = static_cast<int8_t*>(
      checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, false));
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    int8_t* buff,
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(buff);
  CHECK(!storage_);
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(
    const std::vector<int64_t>& target_init_vals) const {
  CHECK(!storage_);
  OOM_TRACE_PUSH(+": size " +
                 std::to_string(query_mem_desc_.getBufferSizeBytes(device_type_)));
  auto buff = static_cast<int8_t*>(
      checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
  storage_.reset(new ResultSetStorage(targets_, query_mem_desc_, buff, false));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

size_t ResultSet::getCurrentRowBufferIndex() const {
  if (crt_row_buff_idx_ == 0) {
    throw std::runtime_error("current row buffer iteration index is undefined");
  }
  return crt_row_buff_idx_ - 1;
}

void ResultSet::append(ResultSet& that) {
  CHECK_EQ(-1, cached_row_count_);
  if (!that.storage_) {
    return;
  }
  appended_storage_.push_back(std::move(that.storage_));
  query_mem_desc_.setEntryCount(
      query_mem_desc_.getEntryCount() +
      appended_storage_.back()->query_mem_desc_.getEntryCount());
  chunks_.insert(chunks_.end(), that.chunks_.begin(), that.chunks_.end());
  col_buffers_.insert(
      col_buffers_.end(), that.col_buffers_.begin(), that.col_buffers_.end());
  frag_offsets_.insert(
      frag_offsets_.end(), that.frag_offsets_.begin(), that.frag_offsets_.end());
  consistent_frag_sizes_.insert(consistent_frag_sizes_.end(),
                                that.consistent_frag_sizes_.begin(),
                                that.consistent_frag_sizes_.end());
  chunk_iters_.insert(
      chunk_iters_.end(), that.chunk_iters_.begin(), that.chunk_iters_.end());
  if (separate_varlen_storage_valid_) {
    CHECK(that.separate_varlen_storage_valid_);
    serialized_varlen_buffer_.insert(serialized_varlen_buffer_.end(),
                                     that.serialized_varlen_buffer_.begin(),
                                     that.serialized_varlen_buffer_.end());
  }
  for (auto& buff : that.literal_buffers_) {
    literal_buffers_.push_back(std::move(buff));
  }
}

const ResultSetStorage* ResultSet::getStorage() const {
  return storage_.get();
}

size_t ResultSet::colCount() const {
  return just_explain_ ? 1 : targets_.size();
}

SQLTypeInfo ResultSet::getColType(const size_t col_idx) const {
  if (just_explain_) {
    return SQLTypeInfo(kTEXT, false);
  }
  CHECK_LT(col_idx, targets_.size());
  return targets_[col_idx].agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false)
                                            : targets_[col_idx].sql_type;
}

size_t ResultSet::rowCount(const bool force_parallel) const {
  if (just_explain_) {
    return 1;
  }
  if (!permutation_.empty()) {
    return permutation_.size();
  }
  if (cached_row_count_ != -1) {
    CHECK_GE(cached_row_count_, 0);
    return cached_row_count_;
  }
  if (!storage_) {
    return 0;
  }
  if (force_parallel || entryCount() > 20000) {
    return parallelRowCount();
  }
  std::lock_guard<std::mutex> lock(row_iteration_mutex_);
  moveToBegin();
  size_t row_count{0};
  while (true) {
    auto crt_row = getNextRowUnlocked(false, false);
    if (crt_row.empty()) {
      break;
    }
    ++row_count;
  }
  moveToBegin();
  return row_count;
}

void ResultSet::setCachedRowCount(const size_t row_count) const {
  CHECK(cached_row_count_ == -1 || cached_row_count_ == static_cast<ssize_t>(row_count));
  cached_row_count_ = row_count;
}

size_t ResultSet::parallelRowCount() const {
  size_t row_count{0};
  const size_t worker_count = cpu_threads();
  std::vector<std::future<size_t>> counter_threads;
  for (size_t i = 0,
              start_entry = 0,
              stride = (entryCount() + worker_count - 1) / worker_count;
       i < worker_count && start_entry < entryCount();
       ++i, start_entry += stride) {
    const auto end_entry = std::min(start_entry + stride, entryCount());
    counter_threads.push_back(std::async(std::launch::async,
                                         [this](const size_t start, const size_t end) {
                                           size_t row_count{0};
                                           for (size_t i = start; i < end; ++i) {
                                             if (!isRowAtEmpty(i)) {
                                               ++row_count;
                                             }
                                           }
                                           return row_count;
                                         },
                                         start_entry,
                                         end_entry));
  }
  for (auto& child : counter_threads) {
    child.wait();
  }
  for (auto& child : counter_threads) {
    row_count += child.get();
  }
  if (keep_first_ + drop_first_) {
    const auto limited_row_count = std::min(keep_first_ + drop_first_, row_count);
    return limited_row_count < drop_first_ ? 0 : limited_row_count - drop_first_;
  }
  return row_count;
}

bool ResultSet::definitelyHasNoRows() const {
  return !storage_ && !estimator_ && !just_explain_;
}

const QueryMemoryDescriptor& ResultSet::getQueryMemDesc() const {
  CHECK(storage_);
  return storage_->query_mem_desc_;
}

const std::vector<TargetInfo>& ResultSet::getTargetInfos() const {
  return targets_;
}

int8_t* ResultSet::getDeviceEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  return estimator_buffer_;
}

int8_t* ResultSet::getHostEstimatorBuffer() const {
  return host_estimator_buffer_;
}

void ResultSet::syncEstimatorBuffer() const {
  CHECK(device_type_ == ExecutorDeviceType::GPU);
  CHECK(!host_estimator_buffer_);
  CHECK_EQ(size_t(0), estimator_->getBufferSize() % sizeof(int64_t));
  OOM_TRACE_PUSH(+": host_estimator_buffer_ " +
                 std::to_string(estimator_->getBufferSize()));
  host_estimator_buffer_ =
      static_cast<int8_t*>(checked_calloc(estimator_->getBufferSize(), 1));
  copy_from_gpu(data_mgr_,
                host_estimator_buffer_,
                reinterpret_cast<CUdeviceptr>(estimator_buffer_),
                estimator_->getBufferSize(),
                device_id_);
}

void ResultSet::setQueueTime(const int64_t queue_time) {
  queue_time_ms_ = queue_time;
}

int64_t ResultSet::getQueueTime() const {
  return queue_time_ms_;
}

int64_t ResultSet::getRenderTime() const {
  return render_time_ms_;
}

void ResultSet::moveToBegin() const {
  crt_row_buff_idx_ = 0;
  fetched_so_far_ = 0;
}

bool ResultSet::isTruncated() const {
  return keep_first_ + drop_first_;
}

QueryMemoryDescriptor ResultSet::fixupQueryMemoryDescriptor(
    const QueryMemoryDescriptor& query_mem_desc) {
  auto query_mem_desc_copy = query_mem_desc;
  query_mem_desc_copy.resetGroupColWidths(
      std::vector<int8_t>(query_mem_desc_copy.groupColWidthsSize(), 8));
  size_t total_bytes{0};
  size_t col_idx = 0;
  for (; col_idx < query_mem_desc_copy.getColCount(); ++col_idx) {
    auto chosen_bytes = query_mem_desc_copy.getColumnWidth(col_idx).compact;
    if (chosen_bytes == sizeof(int64_t)) {
      const auto aligned_total_bytes = align_to_int64(total_bytes);
      CHECK_GE(aligned_total_bytes, total_bytes);
      if (col_idx >= 1) {
        const auto padding = aligned_total_bytes - total_bytes;
        CHECK(padding == 0 || padding == 4);
        query_mem_desc_copy.agg_col_widths_[col_idx - 1].compact += padding;
      }
      total_bytes = aligned_total_bytes;
    }
    total_bytes += chosen_bytes;
  }
  if (!query_mem_desc.sortOnGpu()) {
    const auto aligned_total_bytes = align_to_int64(total_bytes);
    CHECK_GE(aligned_total_bytes, total_bytes);
    const auto padding = aligned_total_bytes - total_bytes;
    CHECK(padding == 0 || padding == 4);
    query_mem_desc_copy.agg_col_widths_[col_idx - 1].compact += padding;
  }
  return query_mem_desc_copy;
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries,
                     const size_t top_n) {
  CHECK_EQ(-1, cached_row_count_);
  CHECK(!targets_.empty());
#ifdef HAVE_CUDA
  if (canUseFastBaselineSort(order_entries, top_n)) {
    baselineSort(order_entries, top_n);
    return;
  }
#endif  // HAVE_CUDA
  if (query_mem_desc_.sortOnGpu()) {
    try {
      OOM_TRACE_PUSH();
      radixSortOnGpu(order_entries);
    } catch (const OutOfMemory&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    } catch (const std::bad_alloc&) {
      LOG(WARNING) << "Out of GPU memory during sort, finish on CPU";
      radixSortOnCpu(order_entries);
    }
    return;
  }
  // This check isn't strictly required, but allows the index buffer to be 32-bit.
  if (query_mem_desc_.getEntryCount() > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }

  CHECK(permutation_.empty());

  const bool use_heap{order_entries.size() == 1 && top_n};
  if (use_heap && entryCount() > 100000) {
    if (g_enable_watchdog && (entryCount() > 20000000)) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    parallelTop(order_entries, top_n);
    return;
  }

  if (g_enable_watchdog && (entryCount() > Executor::baseline_threshold)) {
    throw WatchdogException("Sorting the result would be too slow");
  }

  permutation_ = initPermutationBuffer(0, 1);

  auto compare = createComparator(order_entries, use_heap);

  if (use_heap) {
    topPermutation(permutation_, top_n, compare);
  } else {
    sortPermutation(compare);
  }
}

#ifdef HAVE_CUDA
void ResultSet::baselineSort(const std::list<Analyzer::OrderEntry>& order_entries,
                             const size_t top_n) {
  // If we only have on GPU, it's usually faster to do multi-threaded radix sort on CPU
  if (getGpuCount() > 1) {
    try {
      doBaselineSort(ExecutorDeviceType::GPU, order_entries, top_n);
    } catch (...) {
      doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n);
    }
  } else {
    doBaselineSort(ExecutorDeviceType::CPU, order_entries, top_n);
  }
}
#endif  // HAVE_CUDA

std::vector<uint32_t> ResultSet::initPermutationBuffer(const size_t start,
                                                       const size_t step) {
  CHECK_NE(size_t(0), step);
  std::vector<uint32_t> permutation;
  const auto total_entries = query_mem_desc_.getEntryCount();
  permutation.reserve(total_entries / step);
  for (size_t i = start; i < total_entries; i += step) {
    const auto storage_lookup_result = findStorage(i);
    const auto lhs_storage = storage_lookup_result.storage_ptr;
    const auto off = storage_lookup_result.fixedup_entry_idx;
    CHECK(lhs_storage);
    if (!lhs_storage->isEmptyEntry(off)) {
      permutation.push_back(i);
    }
  }
  return permutation;
}

const std::vector<uint32_t>& ResultSet::getPermutationBuffer() const {
  return permutation_;
}

void ResultSet::parallelTop(const std::list<Analyzer::OrderEntry>& order_entries,
                            const size_t top_n) {
  const size_t step = cpu_threads();
  std::vector<std::vector<uint32_t>> strided_permutations(step);
  std::vector<std::future<void>> init_futures;
  for (size_t start = 0; start < step; ++start) {
    init_futures.emplace_back(
        std::async(std::launch::async, [this, start, step, &strided_permutations] {
          strided_permutations[start] = initPermutationBuffer(start, step);
        }));
  }
  for (auto& init_future : init_futures) {
    init_future.wait();
  }
  for (auto& init_future : init_futures) {
    init_future.get();
  }
  auto compare = createComparator(order_entries, true);
  std::vector<std::future<void>> top_futures;
  for (auto& strided_permutation : strided_permutations) {
    top_futures.emplace_back(
        std::async(std::launch::async, [&strided_permutation, &compare, top_n] {
          topPermutation(strided_permutation, top_n, compare);
        }));
  }
  for (auto& top_future : top_futures) {
    top_future.wait();
  }
  for (auto& top_future : top_futures) {
    top_future.get();
  }
  permutation_.reserve(strided_permutations.size() * top_n);
  for (const auto& strided_permutation : strided_permutations) {
    permutation_.insert(
        permutation_.end(), strided_permutation.begin(), strided_permutation.end());
  }
  topPermutation(permutation_, top_n, compare);
}

std::pair<ssize_t, size_t> ResultSet::getStorageIndex(const size_t entry_idx) const {
  size_t fixedup_entry_idx = entry_idx;
  auto entry_count = storage_->query_mem_desc_.getEntryCount();
  const bool is_rowwise_layout = !storage_->query_mem_desc_.didOutputColumnar();
  if (fixedup_entry_idx < entry_count) {
    return {0, fixedup_entry_idx};
  }
  fixedup_entry_idx -= entry_count;
  for (size_t i = 0; i < appended_storage_.size(); ++i) {
    const auto& desc = appended_storage_[i]->query_mem_desc_;
    CHECK_NE(is_rowwise_layout, desc.didOutputColumnar());
    entry_count = desc.getEntryCount();
    if (fixedup_entry_idx < entry_count) {
      return {i + 1, fixedup_entry_idx};
    }
    fixedup_entry_idx -= entry_count;
  }
  CHECK(false);
  return {-1, entry_idx};
}

ResultSet::StorageLookupResult ResultSet::findStorage(const size_t entry_idx) const {
  ssize_t stg_idx{-1};
  size_t fixedup_entry_idx{entry_idx};
  std::tie(stg_idx, fixedup_entry_idx) = getStorageIndex(entry_idx);
  CHECK_LE(ssize_t(0), stg_idx);
  return {stg_idx ? appended_storage_[stg_idx - 1].get() : storage_.get(),
          fixedup_entry_idx,
          static_cast<size_t>(stg_idx)};
}

template <typename BUFFER_ITERATOR_TYPE>
bool ResultSet::ResultSetComparator<BUFFER_ITERATOR_TYPE>::operator()(
    const uint32_t lhs,
    const uint32_t rhs) const {
  // NB: The compare function must define a strict weak ordering, otherwise
  // std::sort will trigger a segmentation fault (or corrupt memory).
  const auto lhs_storage_lookup_result = result_set_->findStorage(lhs);
  const auto rhs_storage_lookup_result = result_set_->findStorage(rhs);
  const auto lhs_storage = lhs_storage_lookup_result.storage_ptr;
  const auto rhs_storage = rhs_storage_lookup_result.storage_ptr;
  const auto fixedup_lhs = lhs_storage_lookup_result.fixedup_entry_idx;
  const auto fixedup_rhs = rhs_storage_lookup_result.fixedup_entry_idx;
  for (const auto order_entry : order_entries_) {
    CHECK_GE(order_entry.tle_no, 1);
    const auto& agg_info = result_set_->targets_[order_entry.tle_no - 1];
    const auto& entry_ti = get_compact_type(agg_info);
    bool float_argument_input = takes_float_argument(agg_info);
    // Need to determine if the float value has been stored as float
    // or if it has been compacted to a different (often larger 8 bytes)
    // in distributed case the floats are actually 4 bytes
    // TODO the above takes_float_argument() is widely used  wonder if this problem
    // exists elsewhere
    if (entry_ti.get_type() == kFLOAT) {
      const auto is_col_lazy =
          !result_set_->lazy_fetch_info_.empty() &&
          result_set_->lazy_fetch_info_[order_entry.tle_no - 1].is_lazily_fetched;
      if (result_set_->query_mem_desc_.getColumnWidth(order_entry.tle_no - 1).compact ==
              sizeof(float) ||
          (result_set_->query_mem_desc_.didOutputColumnar() && !is_col_lazy &&
           result_set_->query_mem_desc_.getPaddedColumnWidthBytes(order_entry.tle_no -
                                                                  1) == sizeof(float))) {
        float_argument_input = true;
      }
    }
    const auto lhs_v = buffer_itr_.getColumnInternal(lhs_storage->buff_,
                                                     fixedup_lhs,
                                                     order_entry.tle_no - 1,
                                                     lhs_storage_lookup_result);
    const auto rhs_v = buffer_itr_.getColumnInternal(rhs_storage->buff_,
                                                     fixedup_rhs,
                                                     order_entry.tle_no - 1,
                                                     rhs_storage_lookup_result);
    if (UNLIKELY(isNull(entry_ti, lhs_v, float_argument_input) &&
                 isNull(entry_ti, rhs_v, float_argument_input))) {
      return false;
    }
    if (UNLIKELY(isNull(entry_ti, lhs_v, float_argument_input) &&
                 !isNull(entry_ti, rhs_v, float_argument_input))) {
      return use_heap_ ? !order_entry.nulls_first : order_entry.nulls_first;
    }
    if (UNLIKELY(isNull(entry_ti, rhs_v, float_argument_input) &&
                 !isNull(entry_ti, lhs_v, float_argument_input))) {
      return use_heap_ ? order_entry.nulls_first : !order_entry.nulls_first;
    }
    const bool use_desc_cmp = use_heap_ ? !order_entry.is_desc : order_entry.is_desc;
    if (LIKELY(lhs_v.isInt())) {
      CHECK(rhs_v.isInt());
      if (UNLIKELY(entry_ti.is_string() &&
                   entry_ti.get_compression() == kENCODING_DICT)) {
        CHECK_EQ(4, entry_ti.get_logical_size());
        const auto string_dict_proxy = result_set_->executor_->getStringDictionaryProxy(
            entry_ti.get_comp_param(), result_set_->row_set_mem_owner_, false);
        auto lhs_str = string_dict_proxy->getString(lhs_v.i1);
        auto rhs_str = string_dict_proxy->getString(rhs_v.i1);
        if (lhs_str == rhs_str) {
          continue;
        }
        return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
      }
      if (UNLIKELY(is_distinct_target(result_set_->targets_[order_entry.tle_no - 1]))) {
        const auto lhs_sz = count_distinct_set_size(
            lhs_v.i1,
            result_set_->query_mem_desc_.getCountDistinctDescriptor(order_entry.tle_no -
                                                                    1));
        const auto rhs_sz = count_distinct_set_size(
            rhs_v.i1,
            result_set_->query_mem_desc_.getCountDistinctDescriptor(order_entry.tle_no -
                                                                    1));
        if (lhs_sz == rhs_sz) {
          continue;
        }
        return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
      }
      if (lhs_v.i1 == rhs_v.i1) {
        continue;
      }
      if (entry_ti.is_fp()) {
        if (float_argument_input) {
          const auto lhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval = *reinterpret_cast<const float*>(may_alias_ptr(&rhs_v.i1));
          return use_desc_cmp ? lhs_dval > rhs_dval : lhs_dval < rhs_dval;
        } else {
          const auto lhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&lhs_v.i1));
          const auto rhs_dval =
              *reinterpret_cast<const double*>(may_alias_ptr(&rhs_v.i1));
          return use_desc_cmp ? lhs_dval > rhs_dval : lhs_dval < rhs_dval;
        }
      }
      return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
    } else {
      if (lhs_v.isPair()) {
        CHECK(rhs_v.isPair());
        const auto lhs =
            pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti, float_argument_input);
        const auto rhs =
            pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti, float_argument_input);
        if (lhs == rhs) {
          continue;
        }
        return use_desc_cmp ? lhs > rhs : lhs < rhs;
      } else {
        CHECK(lhs_v.isStr() && rhs_v.isStr());
        const auto lhs = lhs_v.strVal();
        const auto rhs = rhs_v.strVal();
        if (lhs == rhs) {
          continue;
        }
        return use_desc_cmp ? lhs > rhs : lhs < rhs;
      }
    }
  }
  return false;
}

void ResultSet::topPermutation(
    std::vector<uint32_t>& to_sort,
    const size_t n,
    const std::function<bool(const uint32_t, const uint32_t)> compare) {
  std::make_heap(to_sort.begin(), to_sort.end(), compare);
  std::vector<uint32_t> permutation_top;
  permutation_top.reserve(n);
  for (size_t i = 0; i < n && !to_sort.empty(); ++i) {
    permutation_top.push_back(to_sort.front());
    std::pop_heap(to_sort.begin(), to_sort.end(), compare);
    to_sort.pop_back();
  }
  to_sort.swap(permutation_top);
}

void ResultSet::sortPermutation(
    const std::function<bool(const uint32_t, const uint32_t)> compare) {
  std::sort(permutation_.begin(), permutation_.end(), compare);
}

void ResultSet::radixSortOnGpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto data_mgr = &executor_->catalog_->getDataMgr();
  const int device_id{0};
  CudaAllocator cuda_allocator(data_mgr, device_id);
  std::vector<int64_t*> group_by_buffers(executor_->blockSize());
  group_by_buffers[0] = reinterpret_cast<int64_t*>(storage_->getUnderlyingBuffer());
  auto dev_group_by_buffers = create_dev_group_by_buffers(cuda_allocator,
                                                          group_by_buffers,
                                                          query_mem_desc_,
                                                          executor_->blockSize(),
                                                          executor_->gridSize(),
                                                          device_id,
                                                          true,
                                                          true,
                                                          nullptr);
  ResultRows::inplaceSortGpuImpl(order_entries,
                                 query_mem_desc_,
                                 GpuQueryMemory{dev_group_by_buffers},
                                 data_mgr,
                                 device_id);
  copy_group_by_buffers_from_gpu(
      data_mgr,
      group_by_buffers,
      query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
      dev_group_by_buffers.second,
      query_mem_desc_,
      executor_->blockSize(),
      executor_->gridSize(),
      device_id,
      false);
}

void ResultSet::radixSortOnCpu(
    const std::list<Analyzer::OrderEntry>& order_entries) const {
  CHECK(!query_mem_desc_.hasKeylessHash());
  std::vector<int64_t> tmp_buff(query_mem_desc_.getEntryCount());
  std::vector<int32_t> idx_buff(query_mem_desc_.getEntryCount());
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = storage_->getUnderlyingBuffer();
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff = reinterpret_cast<int64_t*>(
        buffer_ptr + query_mem_desc_.getColOffInBytes(target_idx));
    const auto chosen_bytes = query_mem_desc_.getColumnWidth(target_idx).compact;
    sort_groups_cpu(sortkey_val_buff,
                    &idx_buff[0],
                    query_mem_desc_.getEntryCount(),
                    order_entry.is_desc,
                    chosen_bytes);
    apply_permutation_cpu(reinterpret_cast<int64_t*>(buffer_ptr),
                          &idx_buff[0],
                          query_mem_desc_.getEntryCount(),
                          &tmp_buff[0],
                          sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.getColCount();
         ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.getColumnWidth(target_idx).compact;
      const auto satellite_val_buff = reinterpret_cast<int64_t*>(
          buffer_ptr + query_mem_desc_.getColOffInBytes(target_idx));
      apply_permutation_cpu(satellite_val_buff,
                            &idx_buff[0],
                            query_mem_desc_.getEntryCount(),
                            &tmp_buff[0],
                            chosen_bytes);
    }
  }
}

void ResultSetStorage::addCountDistinctSetPointerMapping(const int64_t remote_ptr,
                                                         const int64_t ptr) {
  const auto it_ok = count_distinct_sets_mapping_.emplace(remote_ptr, ptr);
  CHECK(it_ok.second);
}

int64_t ResultSetStorage::mappedPtr(const int64_t remote_ptr) const {
  const auto it = count_distinct_sets_mapping_.find(remote_ptr);
  // Due to the removal of completely zero bitmaps in a distributed transfer there will be
  // remote ptr that do not not exists. Return 0 if no pointer found
  if (it == count_distinct_sets_mapping_.end()) {
    return int64_t(0);
  }
  return it->second;
}

size_t ResultSet::getLimit() {
  return keep_first_;
}

bool can_use_parallel_algorithms(const ResultSet& rows) {
  return !rows.isTruncated();
}

bool use_parallel_algorithms(const ResultSet& rows) {
  return can_use_parallel_algorithms(rows) && rows.entryCount() >= 20000;
}
