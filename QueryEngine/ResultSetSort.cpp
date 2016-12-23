/**
 * @file    ResultSetSort.cpp
 * @author  Alex Suhan <alex@mapd.com>
 * @brief   Efficient baseline sort implementation.
 *
 * Copyright (c) 2014 MapD Technologies, Inc.  All rights reserved.
 */

#ifdef HAVE_CUDA
#include "Execute.h"
#include "ResultSet.h"
#include "ResultSetSortImpl.h"

#include "../Shared/thread_count.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <future>

std::unique_ptr<CudaMgr_Namespace::CudaMgr> g_cuda_mgr;  // for unit tests only

namespace {

void set_cuda_context(Data_Namespace::DataMgr* data_mgr, const int device_id) {
  if (data_mgr) {
    data_mgr->cudaMgr_->setContext(device_id);
    return;
  }
  // for unit tests only
  CHECK(g_cuda_mgr);
  g_cuda_mgr->setContext(device_id);
}

}  // namespace

void ResultSet::doBaselineSort(const ExecutorDeviceType device_type,
                               const std::list<Analyzer::OrderEntry>& order_entries,
                               const size_t top_n) {
  CHECK_EQ(size_t(1), order_entries.size());
  CHECK_EQ(size_t(0), query_mem_desc_.entry_count_small);
  CHECK(!query_mem_desc_.output_columnar);
  const auto& oe = order_entries.front();
  CHECK_GT(oe.tle_no, 0);
  CHECK_LE(static_cast<size_t>(oe.tle_no), targets_.size());
  size_t logical_slot_idx = 0;
  size_t physical_slot_idx = 0;
  for (size_t i = 0; i < static_cast<size_t>(oe.tle_no - 1); ++i) {
    if (query_mem_desc_.agg_col_widths[logical_slot_idx].compact) {
      physical_slot_idx = advance_slot(physical_slot_idx, targets_[i]);
    }
    logical_slot_idx = advance_slot(logical_slot_idx, targets_[i]);
  }
  const auto slot_count = get_buffer_col_slot_count(query_mem_desc_);
  const auto key_count = get_groupby_col_count(query_mem_desc_);
  const auto col_off = slot_offset_rowwise(0, physical_slot_idx, key_count, slot_count) * sizeof(int64_t);
  const size_t col_bytes = query_mem_desc_.agg_col_widths[logical_slot_idx].compact;
  const auto row_bytes = get_row_bytes(query_mem_desc_);
  const auto& target_groupby_indices = query_mem_desc_.target_groupby_indices;
  CHECK(target_groupby_indices.empty() || static_cast<size_t>(oe.tle_no) <= target_groupby_indices.size());
  const ssize_t target_groupby_index{target_groupby_indices.empty() ? -1 : target_groupby_indices[oe.tle_no - 1]};
  GroupByBufferLayoutInfo layout{
      query_mem_desc_.entry_count, col_off, col_bytes, row_bytes, targets_[oe.tle_no - 1], target_groupby_index};
  PodOrderEntry pod_oe{oe.tle_no, oe.is_desc, oe.nulls_first};
  auto groupby_buffer = storage_->getUnderlyingBuffer();
  auto data_mgr = getDataManager();
  const auto step = static_cast<size_t>(device_type == ExecutorDeviceType::GPU ? getGpuCount() : cpu_threads());
  CHECK_GE(step, size_t(1));
  if (step > 1) {
    std::vector<std::future<void>> top_futures;
    std::vector<std::vector<uint32_t>> strided_permutations(step);
    for (size_t start = 0; start < step; ++start) {
      top_futures.emplace_back(std::async(
          std::launch::async,
          [&strided_permutations, data_mgr, device_type, groupby_buffer, pod_oe, layout, top_n, start, step] {
            if (device_type == ExecutorDeviceType::GPU) {
              set_cuda_context(data_mgr, start);
            }
            strided_permutations[start] =
                baseline_sort(device_type, start, data_mgr, groupby_buffer, pod_oe, layout, top_n, start, step);
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
      permutation_.insert(permutation_.end(), strided_permutation.begin(), strided_permutation.end());
    }
    auto compare = createComparator(order_entries, true);
    topPermutation(permutation_, top_n, compare);
    return;
  } else {
    permutation_ = baseline_sort(device_type, 0, data_mgr, groupby_buffer, pod_oe, layout, top_n, 0, 1);
  }
}

bool ResultSet::canUseFastBaselineSort(const std::list<Analyzer::OrderEntry>& order_entries, const size_t top_n) {
  if (order_entries.size() != 1) {
    return false;
  }
  const auto& order_entry = order_entries.front();
  CHECK_GE(order_entry.tle_no, 1);
  CHECK_LE(static_cast<size_t>(order_entry.tle_no), targets_.size());
  const auto& target_info = targets_[order_entry.tle_no - 1];
  if (!target_info.sql_type.is_number() || target_info.is_distinct) {
    return false;
  }
  return query_mem_desc_.hash_type == GroupByColRangeType::MultiCol && !query_mem_desc_.getSmallBufferSizeQuad() &&
         top_n;
}

Data_Namespace::DataMgr* ResultSet::getDataManager() const {
  if (executor_) {
    CHECK(executor_->catalog_);
    return &executor_->catalog_->get_dataMgr();
  }
  return nullptr;
}

int ResultSet::getGpuCount() const {
  const auto data_mgr = getDataManager();
  if (!data_mgr) {
    return g_cuda_mgr ? g_cuda_mgr->getDeviceCount() : 0;
  }
  return data_mgr->gpusPresent() ? data_mgr->cudaMgr_->getDeviceCount() : 0;
}
#endif  // HAVE_CUDA
