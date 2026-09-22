/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file    ResultSetSort.cpp
 * @brief   Efficient baseline sort implementation.
 *
 */

#ifdef HAVE_CUDA
#include "Execute.h"
#include "ResultSet.h"
#include "ResultSetSortImpl.h"

#include "../Shared/thread_count.h"

#include <future>

std::unique_ptr<CudaMgr_Namespace::CudaMgr> g_cuda_mgr;  // for unit tests only

namespace {

void set_cuda_context(Data_Namespace::DataMgr* data_mgr, const int device_id) {
  if (data_mgr) {
    data_mgr->getCudaMgr()->setContext(device_id);
    return;
  }
  // for unit tests only
  CHECK(g_cuda_mgr);
  g_cuda_mgr->setContext(device_id);
}

}  // namespace

void ResultSet::doBaselineSort(const ExecutorDeviceType device_type,
                               const std::list<Analyzer::OrderEntry>& order_entries,
                               const size_t top_n,
                               const Executor* executor) {
  CHECK_EQ(size_t(1), order_entries.size());
  CHECK(!query_mem_desc_.didOutputColumnar());
  const auto& oe = order_entries.front();
  CHECK_GT(oe.tle_no, 0);
  CHECK_LE(static_cast<size_t>(oe.tle_no), targets_.size());
  const auto target_idx = static_cast<size_t>(oe.tle_no - 1);
  const auto target_groupby_indices_sz = query_mem_desc_.targetGroupbyIndicesSize();
  CHECK(target_groupby_indices_sz == 0 || target_idx < target_groupby_indices_sz);
  const int64_t target_groupby_index{
      target_groupby_indices_sz == 0 ? -1
                                     : query_mem_desc_.getTargetGroupbyIndex(target_idx)};
  size_t col_off{0};
  size_t col_bytes{0};
  if (target_groupby_index >= 0) {
    col_bytes = query_mem_desc_.getEffectiveKeyWidth();
  } else {
    const auto& target_slots =
        query_mem_desc_.getColSlotContext().getSlotsForCol(target_idx);
    CHECK(!target_slots.empty());
    const auto slot_idx = target_slots.front();
    col_off = query_mem_desc_.getColOffInBytes(slot_idx);
    col_bytes = query_mem_desc_.getPaddedSlotWidthBytes(slot_idx);
  }
  const auto row_bytes = get_row_bytes(query_mem_desc_);
  GroupByBufferLayoutInfo layout{query_mem_desc_.getEntryCount(),
                                 col_off,
                                 col_bytes,
                                 row_bytes,
                                 targets_[oe.tle_no - 1],
                                 target_groupby_index};
  PodOrderEntry pod_oe{oe.tle_no, oe.is_desc, oe.nulls_first};
  auto groupby_buffer = storage_->getUnderlyingBuffer();
  auto data_mgr = getDataManager();
  std::set<int> device_ids_to_sort = executor->getAvailableDevicesToProcessQuery();
  if (device_type == ExecutorDeviceType::CPU) {
    // for CPU execution, we can logically assume thread id represents device id
    device_ids_to_sort.clear();
    for (auto thread_id = 0; thread_id < cpu_threads(); ++thread_id) {
      device_ids_to_sort.insert(thread_id);
    }
  }
  auto const num_devices_to_sort = device_ids_to_sort.size();
  CHECK_GE(num_devices_to_sort, size_t(1));
  const auto key_bytewidth = query_mem_desc_.getEffectiveKeyWidth();
  if (num_devices_to_sort > 1) {
    std::vector<std::future<void>> top_futures;
    std::vector<Permutation> strided_permutations(num_devices_to_sort);
    for (auto it = device_ids_to_sort.begin(); it != device_ids_to_sort.end(); it++) {
      auto const device_id = *it;
      auto const slot_idx = std::distance(device_ids_to_sort.begin(), it);
      top_futures.emplace_back(std::async(
          std::launch::async,
          [&strided_permutations,
           data_mgr,
           device_type,
           groupby_buffer,
           pod_oe,
           key_bytewidth,
           layout,
           top_n,
           device_id,
           slot_idx,
           num_devices_to_sort,
           &executor] {
            CudaAllocator* device_allocator{nullptr};
            CUstream cuda_stream{nullptr};
            if (device_type == ExecutorDeviceType::GPU) {
              set_cuda_context(data_mgr, device_id);
              device_allocator = executor->getCudaAllocator(device_id);
              CHECK(device_allocator);
              cuda_stream = executor->getCudaStream(device_id);
            }
            strided_permutations[slot_idx] =
                (key_bytewidth == 4) ? baseline_sort<int32_t>(device_type,
                                                              device_id,
                                                              data_mgr,
                                                              device_allocator,
                                                              groupby_buffer,
                                                              pod_oe,
                                                              layout,
                                                              top_n,
                                                              slot_idx,
                                                              num_devices_to_sort,
                                                              cuda_stream)
                                     : baseline_sort<int64_t>(device_type,
                                                              device_id,
                                                              data_mgr,
                                                              device_allocator,
                                                              groupby_buffer,
                                                              pod_oe,
                                                              layout,
                                                              top_n,
                                                              slot_idx,
                                                              num_devices_to_sort,
                                                              cuda_stream);
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
    auto pv = PermutationView(permutation_.data(), permutation_.size());
    initMaterializedSortBuffers(order_entries, false, pv.size());
    pv = topPermutation(
        pv, top_n, createComparator(order_entries, pv, executor, false).get());
    if (pv.size() < permutation_.size()) {
      permutation_.resize(pv.size());
      permutation_.shrink_to_fit();
    }
    return;
  } else {
    auto const device_id = *device_ids_to_sort.begin();
    CudaAllocator* device_allocator{nullptr};
    CUstream cuda_stream{nullptr};
    if (device_type == ExecutorDeviceType::GPU) {
      device_allocator = executor->getCudaAllocator(device_id);
      CHECK(device_allocator);
      cuda_stream = executor->getCudaStream(device_id);
    }
    permutation_ = (key_bytewidth == 4) ? baseline_sort<int32_t>(device_type,
                                                                 device_id,
                                                                 data_mgr,
                                                                 device_allocator,
                                                                 groupby_buffer,
                                                                 pod_oe,
                                                                 layout,
                                                                 top_n,
                                                                 /*start=*/0,
                                                                 /*step=*/1,
                                                                 cuda_stream)
                                        : baseline_sort<int64_t>(device_type,
                                                                 device_id,
                                                                 data_mgr,
                                                                 device_allocator,
                                                                 groupby_buffer,
                                                                 pod_oe,
                                                                 layout,
                                                                 top_n,
                                                                 /*start=*/0,
                                                                 /*step=*/1,
                                                                 cuda_stream);
  }
}

bool ResultSet::canUseFastBaselineSort(
    const std::list<Analyzer::OrderEntry>& order_entries,
    const size_t top_n) {
  if (order_entries.size() != 1 || query_mem_desc_.hasKeylessHash() ||
      query_mem_desc_.sortOnGpu() || query_mem_desc_.didOutputColumnar()) {
    return false;
  }
  const auto& order_entry = order_entries.front();
  CHECK_GE(order_entry.tle_no, 1);
  CHECK_LE(static_cast<size_t>(order_entry.tle_no), targets_.size());
  const auto& target_info = targets_[order_entry.tle_no - 1];
  if (!target_info.sql_type.is_number() || is_distinct_target(target_info)) {
    return false;
  }
  return (query_mem_desc_.getQueryDescriptionType() ==
              QueryDescriptionType::GroupByBaselineHash ||
          query_mem_desc_.isSingleColumnGroupByWithPerfectHash()) &&
         top_n;
}

Data_Namespace::DataMgr* ResultSet::getDataManager() const {
  return &Catalog_Namespace::SysCatalog::instance().getDataMgr();
}

int ResultSet::getGpuCount() const {
  const auto data_mgr = getDataManager();
  if (!data_mgr) {
    return g_cuda_mgr ? g_cuda_mgr->getDeviceCount() : 0;
  }
  return data_mgr->gpusPresent() ? data_mgr->getCudaMgr()->getDeviceCount() : 0;
}
#endif  // HAVE_CUDA
