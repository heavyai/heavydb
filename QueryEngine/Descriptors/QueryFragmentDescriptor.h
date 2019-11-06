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
#include <optional>
#include <set>
#include <vector>

#include "../CompilationOptions.h"
#include "Shared/Logger.h"

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

struct ExecutionKernel {
  int device_id;
  FragmentsList fragments;
  std::optional<size_t> outer_tuple_count;
};

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
    for (const auto& device_itr : execution_kernels_per_device_) {
      const auto& execution_kernels = device_itr.second;
      CHECK_EQ(execution_kernels.size(), size_t(1));

      const auto& fragments_list = execution_kernels.front().fragments;
      f(device_itr.first, fragments_list, rowid_lookup_key_);
    }
  }

  template <typename DISPATCH_FCN>
  void assignFragsToKernelDispatch(DISPATCH_FCN f,
                                   const RelAlgExecutionUnit& ra_exe_unit) const {
    for (const auto& device_itr : execution_kernels_per_device_) {
      for (const auto& execution_kernel : device_itr.second) {
        f(device_itr.first, execution_kernel.fragments, rowid_lookup_key_);

        if (terminateDispatchMaybe(ra_exe_unit, execution_kernel)) {
          return;
        }
      }
    }
  }

  bool shouldCheckWorkUnitWatchdog() const {
    return rowid_lookup_key_ < 0 && !execution_kernels_per_device_.empty();
  }

 protected:
  size_t outer_fragments_size_ = 0;
  int64_t rowid_lookup_key_ = -1;

  std::map<int, const TableFragments*> selected_tables_fragments_;

  std::map<int, std::vector<ExecutionKernel>> execution_kernels_per_device_;

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

  bool terminateDispatchMaybe(const RelAlgExecutionUnit& ra_exe_unit,
                              const ExecutionKernel& kernel) const;

  void checkDeviceMemoryUsage(const Fragmenter_Namespace::FragmentInfo& fragment,
                              const int device_id,
                              const size_t num_cols);
};

#endif  // QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H
