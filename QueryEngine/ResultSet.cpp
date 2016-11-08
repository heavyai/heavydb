/**
 * @file    ResultSet.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Basic constructors and methods of the row set interface.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#include "ResultSet.h"
#include "Execute.h"
#include "InPlaceSort.h"
#include "OutputBufferInitialization.h"
#include "RuntimeFunctions.h"
#include "SqlTypesLayout.h"
#include "DataMgr/BufferMgr/BufferMgr.h"
#include "Shared/checked_alloc.h"
#include "Shared/thread_count.h"

#include <algorithm>
#include <bitset>
#include <future>
#include <numeric>

ResultSetStorage::ResultSetStorage(const std::vector<TargetInfo>& targets,
                                   const ExecutorDeviceType device_type,
                                   const QueryMemoryDescriptor& query_mem_desc,
                                   int8_t* buff,
                                   const bool buff_is_provided)
    : targets_(targets), query_mem_desc_(query_mem_desc), buff_(buff), buff_is_provided_(buff_is_provided) {
  for (const auto& target_info : targets_) {
    if (!target_info.sql_type.get_notnull()) {
      int64_t init_val = null_val_bit_pattern(target_info.sql_type);
      target_init_vals_.push_back(target_info.is_agg ? init_val : 0);
    } else {
      target_init_vals_.push_back(target_info.is_agg ? 0xdeadbeef : 0);
    }
    if (target_info.agg_kind == kAVG) {
      target_init_vals_.push_back(0);
    }
  }
}

int8_t* ResultSetStorage::getUnderlyingBuffer() const {
  return buff_;
}

void ResultSet::keepFirstN(const size_t n) {
  keep_first_ = n;
}

void ResultSet::dropFirstN(const size_t n) {
  drop_first_ = n;
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const ExecutorDeviceType device_type,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Executor* executor)
    : targets_(targets),
      device_type_(device_type),
      device_id_(-1),
      query_mem_desc_(query_mem_desc),
      crt_row_buff_idx_(0),
      fetched_so_far_(0),
      drop_first_(0),
      keep_first_(0),
      row_set_mem_owner_(row_set_mem_owner),
      queue_time_ms_(0),
      render_time_ms_(0),
      executor_(executor),
      estimator_buffer_(nullptr),
      host_estimator_buffer_(nullptr),
      data_mgr_(nullptr) {
}

ResultSet::ResultSet(const std::vector<TargetInfo>& targets,
                     const std::vector<ColumnLazyFetchInfo>& lazy_fetch_info,
                     const std::vector<std::vector<const int8_t*>>& col_buffers,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     const QueryMemoryDescriptor& query_mem_desc,
                     const std::shared_ptr<RowSetMemoryOwner> row_set_mem_owner,
                     const Executor* executor)
    : targets_(targets),
      device_type_(device_type),
      device_id_(device_id),
      query_mem_desc_(query_mem_desc),
      crt_row_buff_idx_(0),
      fetched_so_far_(0),
      drop_first_(0),
      keep_first_(0),
      row_set_mem_owner_(row_set_mem_owner),
      queue_time_ms_(0),
      render_time_ms_(0),
      executor_(executor),
      lazy_fetch_info_(lazy_fetch_info),
      col_buffers_(col_buffers),
      estimator_buffer_(nullptr),
      host_estimator_buffer_(nullptr),
      data_mgr_(nullptr) {
}

ResultSet::ResultSet(const std::shared_ptr<const Analyzer::NDVEstimator> estimator,
                     const ExecutorDeviceType device_type,
                     const int device_id,
                     Data_Namespace::DataMgr* data_mgr)
    : device_type_(device_type),
      device_id_(device_id),
      query_mem_desc_{},
      crt_row_buff_idx_(0),
      estimator_(estimator),
      estimator_buffer_(nullptr),
      host_estimator_buffer_(nullptr),
      data_mgr_(data_mgr) {
  std::unique_ptr<int8_t, CheckedAllocDeleter> zeros(
      static_cast<int8_t*>(checked_calloc(estimator_->getEstimatorBufferSize(), 1)));
  if (device_type == ExecutorDeviceType::GPU) {
    estimator_buffer_ =
        reinterpret_cast<int8_t*>(alloc_gpu_mem(data_mgr_, estimator_->getEstimatorBufferSize(), device_id_, nullptr));
    copy_to_gpu(data_mgr,
                reinterpret_cast<CUdeviceptr>(estimator_buffer_),
                zeros.get(),
                estimator_->getEstimatorBufferSize(),
                device_id_);
  } else {
    host_estimator_buffer_ = static_cast<int8_t*>(checked_calloc(estimator_->getEstimatorBufferSize(), 1));
  }
}

ResultSet::ResultSet()
    : device_type_(ExecutorDeviceType::CPU),
      device_id_(-1),
      query_mem_desc_{},
      crt_row_buff_idx_(0),
      estimator_buffer_(nullptr),
      host_estimator_buffer_(nullptr),
      data_mgr_(nullptr) {
}

ResultSet::~ResultSet() {
  if (storage_) {
    CHECK(storage_->getUnderlyingBuffer());
    if (!storage_->buff_is_provided_) {
      free(storage_->getUnderlyingBuffer());
    }
  }
  if (host_estimator_buffer_) {
    CHECK(device_type_ == ExecutorDeviceType::CPU || estimator_buffer_);
    free(host_estimator_buffer_);
  }
}

const ResultSetStorage* ResultSet::allocateStorage() const {
  CHECK(!storage_);
  auto buff = static_cast<int8_t*>(checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
  storage_.reset(new ResultSetStorage(targets_, device_type_, query_mem_desc_, buff, false));
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(int8_t* buff, const std::vector<int64_t>& target_init_vals) const {
  CHECK(buff);
  storage_.reset(new ResultSetStorage(targets_, device_type_, query_mem_desc_, buff, true));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

const ResultSetStorage* ResultSet::allocateStorage(const std::vector<int64_t>& target_init_vals) const {
  CHECK(!storage_);
  auto buff = static_cast<int8_t*>(checked_malloc(query_mem_desc_.getBufferSizeBytes(device_type_)));
  storage_.reset(new ResultSetStorage(targets_, device_type_, query_mem_desc_, buff, false));
  storage_->target_init_vals_ = target_init_vals;
  return storage_.get();
}

void ResultSet::append(ResultSet& that) {
  CHECK(!query_mem_desc_.output_columnar);  // TODO(miyu)
  if (!that.storage_) {
    return;
  }
  appended_storage_.push_back(std::move(that.storage_));
  query_mem_desc_.entry_count += appended_storage_.back()->query_mem_desc_.entry_count;
  query_mem_desc_.entry_count_small += appended_storage_.back()->query_mem_desc_.entry_count_small;
  chunks_.insert(chunks_.end(), that.chunks_.begin(), that.chunks_.end());
  col_buffers_.insert(col_buffers_.end(), that.col_buffers_.begin(), that.col_buffers_.end());
  chunk_iters_.insert(chunk_iters_.end(), that.chunk_iters_.begin(), that.chunk_iters_.end());
  for (auto& buff : that.literal_buffers_) {
    literal_buffers_.push_back(std::move(buff));
  }
}

const ResultSetStorage* ResultSet::getStorage() const {
  return storage_.get();
}

size_t ResultSet::colCount() const {
  return targets_.size();
}

SQLTypeInfo ResultSet::getColType(const size_t col_idx) const {
  CHECK_LT(col_idx, targets_.size());
  return targets_[col_idx].agg_kind == kAVG ? SQLTypeInfo(kDOUBLE, false) : targets_[col_idx].sql_type;
}

size_t ResultSet::rowCount() const {
  moveToBegin();
  size_t row_count{0};
  while (true) {
    auto crt_row = getNextRow(false, false);
    if (crt_row.empty()) {
      break;
    }
    ++row_count;
  }
  moveToBegin();
  return row_count;
}

bool ResultSet::definitelyHasNoRows() const {
  return !storage_ && !estimator_;
}

const QueryMemoryDescriptor& ResultSet::getQueryMemDesc() const {
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
  CHECK_EQ(size_t(0), estimator_->getEstimatorBufferSize() % sizeof(int64_t));
  host_estimator_buffer_ = static_cast<int8_t*>(checked_calloc(estimator_->getEstimatorBufferSize(), 1));
  copy_from_gpu(data_mgr_,
                host_estimator_buffer_,
                reinterpret_cast<CUdeviceptr>(estimator_buffer_),
                estimator_->getEstimatorBufferSize(),
                device_id_);
}

size_t ResultSet::getNDVEstimator() const {
  CHECK(host_estimator_buffer_);
  size_t bits_set = 0;
  for (size_t i = 0; i < estimator_->getEstimatorBufferSize(); ++i) {
    const auto bitfield = host_estimator_buffer_[i];
    bits_set += std::bitset<8>(bitfield).count();
  }
  const auto total_bits = estimator_->getEstimatorBufferSize() * 8;
  CHECK_LE(bits_set, total_bits);
  const auto unset_bits = total_bits - bits_set;
  const auto ratio = static_cast<double>(unset_bits) / total_bits;
  if (ratio == 0.) {
    return total_bits;
  }
  return -static_cast<double>(total_bits) * log(ratio);
}

void ResultSet::setQueueTime(const int64_t queue_time) {
  queue_time_ms_ = queue_time;
}

int64_t ResultSet::getQueueTime() const {
  return queue_time_ms_;
}

void ResultSet::moveToBegin() const {
  crt_row_buff_idx_ = 0;
  fetched_so_far_ = 0;
}

QueryMemoryDescriptor ResultSet::fixupQueryMemoryDescriptor(const QueryMemoryDescriptor& query_mem_desc) {
  auto query_mem_desc_copy = query_mem_desc;
  for (auto& group_width : query_mem_desc_copy.group_col_widths) {
    group_width = 8;
  }
  size_t total_bytes{0};
  size_t col_idx = 0;
  for (; col_idx < query_mem_desc_copy.agg_col_widths.size(); ++col_idx) {
    auto chosen_bytes = query_mem_desc_copy.agg_col_widths[col_idx].compact;
    if (chosen_bytes == sizeof(int64_t)) {
      const auto aligned_total_bytes = align_to_int64(total_bytes);
      CHECK_GE(aligned_total_bytes, total_bytes);
      if (col_idx >= 1) {
        const auto padding = aligned_total_bytes - total_bytes;
        CHECK(padding == 0 || padding == 4);
        query_mem_desc_copy.agg_col_widths[col_idx - 1].compact += padding;
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
    query_mem_desc_copy.agg_col_widths[col_idx - 1].compact += padding;
  }
  if (query_mem_desc_copy.entry_count_small > 0) {
    query_mem_desc_copy.hash_type = GroupByColRangeType::OneColGuessedRange;
  }
  return query_mem_desc_copy;
}

void ResultSet::sort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n) {
  if (isEmptyInitializer()) {
    return;
  }
  if (query_mem_desc_.sortOnGpu()) {
    try {
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
  if (query_mem_desc_.entry_count > std::numeric_limits<uint32_t>::max()) {
    throw RowSortException("Sorting more than 4B elements not supported");
  }

  CHECK(size_t(0) == query_mem_desc_.entry_count_small || !query_mem_desc_.output_columnar);  // TODO(alex)
  CHECK(permutation_.empty());

  const bool use_heap{order_entries.size() == 1 && top_n};
  if (use_heap && entryCount() > 100000) {
    if (g_enable_watchdog && (entryCount() > 20000000)) {
      throw WatchdogException("Sorting the result would be too slow");
    }
    parallelTop(order_entries, top_n);
    return;
  }

  if (g_enable_watchdog && (entryCount() > 100000)) {
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

std::vector<uint32_t> ResultSet::initPermutationBuffer(const size_t start, const size_t step) {
  CHECK_NE(size_t(0), step);
  std::vector<uint32_t> permutation;
  const auto total_entries = query_mem_desc_.entry_count + query_mem_desc_.entry_count_small;
  permutation.reserve(total_entries / step);
  for (size_t i = start; i < query_mem_desc_.entry_count + query_mem_desc_.entry_count_small; i += step) {
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

void ResultSet::parallelTop(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n) {
  const size_t step = cpu_threads();
  strided_permutations_.resize(step);
  std::vector<std::future<void>> init_futures;
  for (size_t start = 0; start < step; ++start) {
    init_futures.emplace_back(std::async(std::launch::async, [this, start, step] {
      strided_permutations_[start] = initPermutationBuffer(start, step);
    }));
  }
  for (auto& init_future : init_futures) {
    init_future.get();
  }
  auto compare = createComparator(order_entries, true);
  std::vector<std::future<void>> top_futures;
  for (auto& strided_permutation : strided_permutations_) {
    top_futures.emplace_back(std::async(std::launch::async, [&strided_permutation, &compare, top_n] {
      topPermutation(strided_permutation, top_n, compare);
    }));
  }
  for (auto& top_future : top_futures) {
    top_future.get();
  }
  permutation_.reserve(strided_permutations_.size() * top_n);
  for (const auto& strided_permutation : strided_permutations_) {
    permutation_.insert(permutation_.end(), strided_permutation.begin(), strided_permutation.end());
  }
  topPermutation(permutation_, top_n, compare);
}

std::pair<ssize_t, size_t> ResultSet::getStorageIndex(const size_t entry_idx) const {
  size_t fixedup_entry_idx = entry_idx;
  auto entry_count = storage_->query_mem_desc_.entry_count;
  const bool is_rowwise_layout = !storage_->query_mem_desc_.output_columnar;
  if (is_rowwise_layout) {
    entry_count += storage_->query_mem_desc_.entry_count_small;
  }
  if (fixedup_entry_idx < entry_count) {
    return {0, fixedup_entry_idx};
  }
  fixedup_entry_idx -= entry_count;
  for (size_t i = 0; i < appended_storage_.size(); ++i) {
    const auto desc = appended_storage_[i]->query_mem_desc_;
    CHECK_NE(is_rowwise_layout, desc.output_columnar);
    entry_count = desc.entry_count;
    if (is_rowwise_layout) {
      entry_count += desc.entry_count_small;
    }
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
  return {
      stg_idx ? appended_storage_[stg_idx - 1].get() : storage_.get(), fixedup_entry_idx, static_cast<size_t>(stg_idx)};
}

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

std::function<bool(const uint32_t, const uint32_t)> ResultSet::createComparator(
    const std::list<Analyzer::OrderEntry>& order_entries,
    const bool use_heap) const {
  return [this, &order_entries, use_heap](const uint32_t lhs, const uint32_t rhs) {
    // NB: The compare function must define a strict weak ordering, otherwise
    // std::sort will trigger a segmentation fault (or corrupt memory).
    const auto lhs_storage_lookup_result = findStorage(lhs);
    const auto rhs_storage_lookup_result = findStorage(rhs);
    const auto lhs_storage = lhs_storage_lookup_result.storage_ptr;
    const auto rhs_storage = rhs_storage_lookup_result.storage_ptr;
    const auto fixedup_lhs = lhs_storage_lookup_result.fixedup_entry_idx;
    const auto fixedup_rhs = rhs_storage_lookup_result.fixedup_entry_idx;
    for (const auto order_entry : order_entries) {
      CHECK_GE(order_entry.tle_no, 1);
      const auto& entry_ti = get_compact_type(targets_[order_entry.tle_no - 1]);
      const auto lhs_v =
          getColumnInternal(lhs_storage->buff_, fixedup_lhs, order_entry.tle_no - 1, lhs_storage_lookup_result);
      const auto rhs_v =
          getColumnInternal(rhs_storage->buff_, fixedup_rhs, order_entry.tle_no - 1, rhs_storage_lookup_result);
      if (UNLIKELY(isNull(entry_ti, lhs_v) && isNull(entry_ti, rhs_v))) {
        return false;
      }
      if (UNLIKELY(isNull(entry_ti, lhs_v) && !isNull(entry_ti, rhs_v))) {
        return use_heap ? !order_entry.nulls_first : order_entry.nulls_first;
      }
      if (UNLIKELY(isNull(entry_ti, rhs_v) && !isNull(entry_ti, lhs_v))) {
        return use_heap ? order_entry.nulls_first : !order_entry.nulls_first;
      }
      const bool use_desc_cmp = use_heap ? !order_entry.is_desc : order_entry.is_desc;
      if (LIKELY(lhs_v.isInt())) {
        CHECK(rhs_v.isInt());
        if (UNLIKELY(entry_ti.is_string() && entry_ti.get_compression() == kENCODING_DICT)) {
          CHECK_EQ(4, entry_ti.get_logical_size());
          const auto string_dict = executor_->getStringDictionary(entry_ti.get_comp_param(), row_set_mem_owner_);
          auto lhs_str = string_dict->getString(lhs_v.i1);
          auto rhs_str = string_dict->getString(rhs_v.i1);
          if (lhs_str == rhs_str) {
            continue;
          }
          return use_desc_cmp ? lhs_str > rhs_str : lhs_str < rhs_str;
        }
        if (UNLIKELY(targets_[order_entry.tle_no - 1].is_distinct)) {
          const auto lhs_sz =
              bitmap_set_size(lhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->getCountDistinctDescriptors());
          const auto rhs_sz =
              bitmap_set_size(rhs_v.i1, order_entry.tle_no - 1, row_set_mem_owner_->getCountDistinctDescriptors());
          if (lhs_sz == rhs_sz) {
            continue;
          }
          return use_desc_cmp ? lhs_sz > rhs_sz : lhs_sz < rhs_sz;
        }
        if (lhs_v.i1 == rhs_v.i1) {
          continue;
        }
        return use_desc_cmp ? lhs_v.i1 > rhs_v.i1 : lhs_v.i1 < rhs_v.i1;
      } else {
        if (lhs_v.isPair()) {
          CHECK(rhs_v.isPair());
          const auto lhs = pair_to_double({lhs_v.i1, lhs_v.i2}, entry_ti);
          const auto rhs = pair_to_double({rhs_v.i1, rhs_v.i2}, entry_ti);
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
  };
}

#undef UNLIKELY
#undef LIKELY

void ResultSet::topPermutation(std::vector<uint32_t>& to_sort,
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

void ResultSet::sortPermutation(const std::function<bool(const uint32_t, const uint32_t)> compare) {
  std::sort(permutation_.begin(), permutation_.end(), compare);
}

bool ResultSet::isEmptyInitializer() const {
  return targets_.empty();
}

void ResultSet::radixSortOnGpu(const std::list<Analyzer::OrderEntry>& order_entries) const {
  auto data_mgr = &executor_->catalog_->get_dataMgr();
  const int device_id{0};
  std::vector<int64_t*> group_by_buffers(executor_->blockSize());
  group_by_buffers[0] = reinterpret_cast<int64_t*>(storage_->getUnderlyingBuffer());
  auto gpu_query_mem = create_dev_group_by_buffers(data_mgr,
                                                   group_by_buffers,
                                                   {},
                                                   query_mem_desc_,
                                                   executor_->blockSize(),
                                                   executor_->gridSize(),
                                                   device_id,
                                                   true,
                                                   true,
                                                   nullptr);
  size_t max_entry_size{0};
  for (const auto& wid : query_mem_desc_.agg_col_widths) {
    max_entry_size = std::max(max_entry_size, size_t(wid.compact));
  }
  if (!query_mem_desc_.keyless_hash) {
    max_entry_size = std::max(max_entry_size, sizeof(int64_t));
  }
  ScopedScratchBuffer scratch_buff(query_mem_desc_.entry_count * max_entry_size, data_mgr, device_id);
  ResultRows::inplaceSortGpuImpl(
      order_entries, query_mem_desc_, gpu_query_mem, reinterpret_cast<int64_t*>(scratch_buff.getPtr()));
  copy_group_by_buffers_from_gpu(data_mgr,
                                 group_by_buffers,
                                 query_mem_desc_.getBufferSizeBytes(ExecutorDeviceType::GPU),
                                 gpu_query_mem.group_by_buffers.second,
                                 query_mem_desc_,
                                 executor_->blockSize(),
                                 executor_->gridSize(),
                                 device_id,
                                 false);
}

void ResultSet::radixSortOnCpu(const std::list<Analyzer::OrderEntry>& order_entries) const {
  CHECK(!query_mem_desc_.keyless_hash);
  std::vector<int64_t> tmp_buff(query_mem_desc_.entry_count);
  std::vector<int32_t> idx_buff(query_mem_desc_.entry_count);
  CHECK_EQ(size_t(1), order_entries.size());
  auto buffer_ptr = storage_->getUnderlyingBuffer();
  for (const auto& order_entry : order_entries) {
    const auto target_idx = order_entry.tle_no - 1;
    const auto sortkey_val_buff =
        reinterpret_cast<int64_t*>(buffer_ptr + query_mem_desc_.getColOffInBytes(0, target_idx));
    const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
    sort_groups_cpu(sortkey_val_buff, &idx_buff[0], query_mem_desc_.entry_count, order_entry.is_desc, chosen_bytes);
    apply_permutation_cpu(reinterpret_cast<int64_t*>(buffer_ptr),
                          &idx_buff[0],
                          query_mem_desc_.entry_count,
                          &tmp_buff[0],
                          sizeof(int64_t));
    for (size_t target_idx = 0; target_idx < query_mem_desc_.agg_col_widths.size(); ++target_idx) {
      if (static_cast<int>(target_idx) == order_entry.tle_no - 1) {
        continue;
      }
      const auto chosen_bytes = query_mem_desc_.agg_col_widths[target_idx].compact;
      const auto satellite_val_buff =
          reinterpret_cast<int64_t*>(buffer_ptr + query_mem_desc_.getColOffInBytes(0, target_idx));
      apply_permutation_cpu(satellite_val_buff, &idx_buff[0], query_mem_desc_.entry_count, &tmp_buff[0], chosen_bytes);
    }
  }
}
