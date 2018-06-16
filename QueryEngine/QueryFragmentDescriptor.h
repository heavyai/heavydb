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
 * @file    QueryMemoryDescriptor.h
 * @author  Alex Baden <alex.baden@mapd.com>
 * @brief   Descriptor for the fragments required for a query.
 */

#ifndef QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H
#define QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H

#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "CompilationOptions.h"

namespace Fragmenter_Namespace {
class FragmentInfo;
}

class Executor;
class InputTableInfo;
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
                          Executor* executor);

  void buildFragmentDeviceMap(const RelAlgExecutionUnit& ra_exe_unit,
                              const std::vector<uint64_t>& frag_offsets,
                              const int device_count,
                              const ExecutorDeviceType& device_type,
                              const bool enable_multifrag_kernels,
                              const bool enable_inner_join_fragment_skipping);

  template <typename FUNCTOR_TYPE>
  void assignFragsToMultiDispatch(FUNCTOR_TYPE f) const {
    for (const auto& kv : fragments_per_device_) {
      f(kv.first, kv.second, rowid_lookup_key_);
    }
  }

  template <typename FUNCTOR_TYPE>
  void assignFragsToDispatch(FUNCTOR_TYPE f) const {
    f(fragments_per_device_, outer_fragments_to_device_id_, rowid_lookup_key_);
  }

  int64_t getRowIdLookupKey() const { return rowid_lookup_key_; }

  size_t getOuterFragmentsSize() const { return outer_fragments_size_; }

  const std::pair<bool, std::shared_ptr<const FragmentsList>> getFragListForIndex(const size_t i) const {
    const auto frag_itr = fragments_per_device_.find(i);
    if (frag_itr == fragments_per_device_.end()) {
      return std::make_pair(false, nullptr);
    } else {
      return std::make_pair(true, std::make_shared<const FragmentsList>(frag_itr->second));
    }
  }

  const int getDeviceForFragment(const size_t i) const {
    const auto device_itr = outer_fragments_to_device_id_.find(i);
    if (device_itr == outer_fragments_to_device_id_.end()) {
      return -1;
    } else {
      return device_itr->second;
    }
  }

  const size_t getOuterFragmentTupleSize(const size_t frag_index) const {
    if (frag_index < outer_fragment_tuple_sizes_.size()) {
      return outer_fragment_tuple_sizes_[frag_index];
    } else {
      return 0;
    }
  }

 protected:
  size_t outer_fragments_size_ = 0;
  int64_t rowid_lookup_key_ = -1;

  std::map<int, const TableFragments*> selected_tables_fragments_;
  std::unordered_map<int, FragmentsList> fragments_per_device_;
  std::unordered_map<int, int> outer_fragments_to_device_id_;
  std::vector<size_t> outer_fragment_tuple_sizes_;

  Executor* executor_;

  void buildFragmentPerKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                                 const std::vector<uint64_t>& frag_offsets,
                                 const int device_count,
                                 const ExecutorDeviceType& device_type);

  void buildMultifragKernelMap(const RelAlgExecutionUnit& ra_exe_unit,
                               const std::vector<uint64_t>& frag_offsets,
                               const int device_count,
                               const ExecutorDeviceType& device_type,
                               const bool enable_inner_join_fragment_skipping);
};

#endif  // QUERYENGINE_QUERYFRAGMENTDESCRIPTOR_H
