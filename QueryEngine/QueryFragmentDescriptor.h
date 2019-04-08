/*
 * Copyright 2018 MapD Technologies, Inc.
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
 * @file    QueryFragmentDescriptor.h
 * @author  Alex Baden <alex.baden@mapd.com>
 * @brief   Descriptor for the fragments required for a query.
 */

#ifndef QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H
#define QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H

#include <deque>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include <glog/logging.h>
#include "CompilationOptions.h"

namespace Fragmenter_Namespace {
class FragmentInfo;
}

namespace Data_Namespace {
struct MemoryInfo;
}

class Executor;
struct InputTableInfo;
struct RelAlgExecutionUnit;

struct FragmentsPerTable {
  int table_id;
  std::vector<size_t> fragment_ids;
};

using FragmentsList = std::vector<FragmentsPerTable>;
using TableFragments = std::deque<Fragmenter_Namespace::FragmentInfo>;

class QueryFragmentDescriptor {
 public:
  QueryFragmentDescriptor(const RelAlgExecutionUnit& ra_exe_unit,
                          const std::vector<InputTableInfo>& query_infos,
                          const std::vector<Data_Namespace::MemoryInfo>& gpu_mem_infos,
                          const double gpu_input_mem_limit_percent);

  static void computeAllTablesFragments(
      std::map<int, const TableFragments*>& all_tables_fragments,
      const RelAlgExecutionUnit& ra_exe_unit,
      const std::vector<InputTableInfo>& query_infos);

  void buildFragmentKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                              const std::vector<uint64_t>& frag_offsets,
                              const int device_count,
                              const ExecutorDeviceType& device_type,
                              const bool enable_multifrag_kernels,
                              const bool enable_inner_join_fragment_skipping,
                              Executor* executor);

  template <typename DISPATCH_FCN>
  void assignFragsToMultiDispatch(DISPATCH_FCN f) const {
    for (const auto& kv : kernels_per_device_) {
      CHECK_EQ(kv.second.size(), size_t(1));
      const auto kernel_id = *kv.second.begin();
      CHECK_LT(kernel_id, fragments_per_kernel_.size());

      f(kv.first, fragments_per_kernel_[kernel_id], rowid_lookup_key_);
    }
  }

  template <typename DISPATCH_FCN>
  void assignFragsToKernelDispatch(DISPATCH_FCN f,
                                   const RelAlgExecutionUnit& ra_exe_unit) const {
    for (const auto& kv : kernels_per_device_) {
      for (const auto& kernel_id : kv.second) {
        CHECK_LT(kernel_id, fragments_per_kernel_.size());

        const auto frag_list = fragments_per_kernel_[kernel_id];
        f(kv.first, frag_list, rowid_lookup_key_);

        if (terminateDispatchMaybe(ra_exe_unit, kernel_id)) {
          return;
        }
      }
    }
  }

  bool shouldCheckWorkUnitWatchdog() const {
    return rowid_lookup_key_ < 0 && fragments_per_kernel_.size() > 0;
  }

 protected:
  size_t outer_fragments_size_ = 0;
  int64_t rowid_lookup_key_ = -1;

  std::map<int, const TableFragments*> selected_tables_fragments_;

  std::vector<FragmentsList> fragments_per_kernel_;
  std::map<int, std::set<size_t>> kernels_per_device_;
  std::vector<size_t> outer_fragment_tuple_sizes_;

  double gpu_input_mem_limit_percent_;
  std::map<size_t, size_t> tuple_count_per_device_;
  std::map<size_t, size_t> available_gpu_mem_bytes_;

  void buildFragmentPerKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<uint64_t>& frag_offsets,
                                 const int device_count,
                                 const ExecutorDeviceType& device_type,
                                 Executor* executor);

  void buildMultifragKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<uint64_t>& frag_offsets,
                               const int device_count,
                               const ExecutorDeviceType& device_type,
                               const bool enable_inner_join_fragment_skipping,
                               Executor* executor);

  const size_t getOuterFragmentTupleSize(const size_t frag_index) const {
    if (frag_index < outer_fragment_tuple_sizes_.size()) {
      return outer_fragment_tuple_sizes_[frag_index];
    } else {
      return 0;
    }
  }

  bool terminateDispatchMaybe(const RelAlgExecutionUnit& ra_exe_unit,
                              const size_t kernel_id) const;

  void checkDeviceMemoryUsage(const Fragmenter_Namespace::FragmentInfo& fragment,
                              const int device_id,
                              const size_t num_cols);
};

#endif  // QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H
