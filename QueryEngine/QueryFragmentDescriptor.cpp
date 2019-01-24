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

#include "QueryFragmentDescriptor.h"

#include <DataMgr/DataMgr.h>
#include "Execute.h"

QueryFragmentDescriptor::QueryFragmentDescriptor(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos,
    const std::vector<Data_Namespace::MemoryInfo>& gpu_mem_infos,
    const double gpu_input_mem_limit_percent)
    : gpu_input_mem_limit_percent_(gpu_input_mem_limit_percent) {
  const size_t input_desc_count{ra_exe_unit.input_descs.size()};
  CHECK_EQ(query_infos.size(), input_desc_count);
  for (size_t table_idx = 0; table_idx < input_desc_count; ++table_idx) {
    const auto table_id = ra_exe_unit.input_descs[table_idx].getTableId();
    if (!selected_tables_fragments_.count(table_id)) {
      selected_tables_fragments_[ra_exe_unit.input_descs[table_idx].getTableId()] =
          &query_infos[table_idx].info.fragments;
    }
  }

  for (size_t device_id = 0; device_id < gpu_mem_infos.size(); device_id++) {
    const auto& gpu_mem_info = gpu_mem_infos[device_id];
    available_gpu_mem_bytes_[device_id] =
        gpu_mem_info.maxNumPages * gpu_mem_info.pageSize;
  }
}

void QueryFragmentDescriptor::computeAllTablesFragments(
    std::map<int, const TableFragments*>& all_tables_fragments,
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<InputTableInfo>& query_infos) {
  for (size_t tab_idx = 0; tab_idx < ra_exe_unit.input_descs.size(); ++tab_idx) {
    int table_id = ra_exe_unit.input_descs[tab_idx].getTableId();
    CHECK_EQ(query_infos[tab_idx].table_id, table_id);
    const auto& fragments = query_infos[tab_idx].info.fragments;
    if (!all_tables_fragments.count(table_id)) {
      all_tables_fragments.insert(std::make_pair(table_id, &fragments));
    }
  }
}

void QueryFragmentDescriptor::buildFragmentKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type,
    const bool enable_multifrag_kernels,
    const bool enable_inner_join_fragment_skipping,
    Executor* executor) {
  if (enable_multifrag_kernels) {
    buildMultifragKernelMap(ra_exe_unit,
                            frag_offsets,
                            device_count,
                            device_type,
                            enable_inner_join_fragment_skipping,
                            executor);
  } else {
    buildFragmentPerKernelMap(
        ra_exe_unit, frag_offsets, device_count, device_type, executor);
  }
}

void QueryFragmentDescriptor::buildFragmentPerKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type,
    Executor* executor) {
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments_.find(outer_table_id);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  const auto num_bytes_for_row = executor->getNumBytesForFetchedRow();

  for (size_t i = 0; i < outer_fragments->size(); ++i) {
    const auto& fragment = (*outer_fragments)[i];
    const auto skip_frag = executor->skipFragment(
        outer_table_desc, fragment, ra_exe_unit.simple_quals, frag_offsets, i);
    if (skip_frag.first) {
      continue;
    }
    // NOTE: Using kernel index instead of frag index now
    outer_fragment_tuple_sizes_.push_back(fragment.getNumTuples());
    rowid_lookup_key_ = std::max(rowid_lookup_key_, skip_frag.second);
    const int chosen_device_count =
        device_type == ExecutorDeviceType::CPU ? 1 : device_count;
    CHECK_GT(chosen_device_count, 0);
    const auto memory_level = device_type == ExecutorDeviceType::GPU
                                  ? Data_Namespace::GPU_LEVEL
                                  : Data_Namespace::CPU_LEVEL;
    int device_id = (device_type == ExecutorDeviceType::CPU || fragment.shard == -1)
                        ? fragment.deviceIds[static_cast<int>(memory_level)]
                        : fragment.shard % chosen_device_count;

    if (device_type == ExecutorDeviceType::GPU) {
      checkDeviceMemoryUsage(fragment, device_id, num_bytes_for_row);
    }
    // Since we may have skipped fragments, the fragments_per_kernel_ vector may be
    // smaller than the outer_fragments size
    CHECK_LE(fragments_per_kernel_.size(), i);
    fragments_per_kernel_.emplace_back(FragmentsList{});
    const auto kernel_id = fragments_per_kernel_.size() - 1;
    CHECK(kernels_per_device_[device_id].insert(kernel_id).second);

    for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
      const auto frag_ids =
          executor->getTableFragmentIndices(ra_exe_unit,
                                            device_type,
                                            j,
                                            i,
                                            selected_tables_fragments_,
                                            executor->getInnerTabIdToJoinCond());
      const auto table_id = ra_exe_unit.input_descs[j].getTableId();
      auto table_frags_it = selected_tables_fragments_.find(table_id);
      CHECK(table_frags_it != selected_tables_fragments_.end());

      fragments_per_kernel_[kernel_id].emplace_back(
          FragmentsPerTable{table_id, frag_ids});
    }
  }
}

void QueryFragmentDescriptor::buildMultifragKernelMap(
    const RelAlgExecutionUnit& ra_exe_unit,
    const std::vector<uint64_t>& frag_offsets,
    const int device_count,
    const ExecutorDeviceType& device_type,
    const bool enable_inner_join_fragment_skipping,
    Executor* executor) {
  // Allocate all the fragments of the tables involved in the query to available
  // devices. The basic idea: the device is decided by the outer table in the
  // query (the first table in a join) and we need to broadcast the fragments
  // in the inner table to each device. Sharding will change this model.
  const auto& outer_table_desc = ra_exe_unit.input_descs.front();
  const int outer_table_id = outer_table_desc.getTableId();
  auto it = selected_tables_fragments_.find(outer_table_id);
  CHECK(it != selected_tables_fragments_.end());
  const auto outer_fragments = it->second;
  outer_fragments_size_ = outer_fragments->size();

  const auto inner_table_id_to_join_condition = executor->getInnerTabIdToJoinCond();
  const auto num_bytes_for_row = executor->getNumBytesForFetchedRow();

  for (size_t outer_frag_id = 0; outer_frag_id < outer_fragments->size();
       ++outer_frag_id) {
    const auto& fragment = (*outer_fragments)[outer_frag_id];
    auto skip_frag = executor->skipFragment(outer_table_desc,
                                            fragment,
                                            ra_exe_unit.simple_quals,
                                            frag_offsets,
                                            outer_frag_id);
    if (enable_inner_join_fragment_skipping &&
        (skip_frag == std::pair<bool, int64_t>(false, -1))) {
      skip_frag = executor->skipFragmentInnerJoins(
          outer_table_desc, ra_exe_unit, fragment, frag_offsets, outer_frag_id);
    }
    if (skip_frag.first) {
      continue;
    }
    const int device_id =
        fragment.shard == -1
            ? fragment.deviceIds[static_cast<int>(Data_Namespace::GPU_LEVEL)]
            : fragment.shard % device_count;
    if (device_type == ExecutorDeviceType::GPU) {
      checkDeviceMemoryUsage(fragment, device_id, num_bytes_for_row);
    }
    for (size_t j = 0; j < ra_exe_unit.input_descs.size(); ++j) {
      const auto table_id = ra_exe_unit.input_descs[j].getTableId();
      auto table_frags_it = selected_tables_fragments_.find(table_id);
      CHECK(table_frags_it != selected_tables_fragments_.end());
      const auto frag_ids =
          executor->getTableFragmentIndices(ra_exe_unit,
                                            device_type,
                                            j,
                                            outer_frag_id,
                                            selected_tables_fragments_,
                                            inner_table_id_to_join_condition);

      if (kernels_per_device_.find(device_id) == kernels_per_device_.end()) {
        fragments_per_kernel_.emplace_back(FragmentsList{});
        kernels_per_device_.insert(std::make_pair(
            device_id, std::set<size_t>({fragments_per_kernel_.size() - 1})));
      }
      const auto kernel_id = *kernels_per_device_[device_id].begin();
      CHECK_LT(kernel_id, fragments_per_kernel_.size());
      if (fragments_per_kernel_[kernel_id].size() < j + 1) {
        fragments_per_kernel_[kernel_id].emplace_back(
            FragmentsPerTable{table_id, frag_ids});
      } else {
        CHECK_EQ(fragments_per_kernel_[kernel_id][j].table_id, table_id);
        auto& curr_frag_ids = fragments_per_kernel_[kernel_id][j].fragment_ids;
        for (const int frag_id : frag_ids) {
          if (std::find(curr_frag_ids.begin(), curr_frag_ids.end(), frag_id) ==
              curr_frag_ids.end()) {
            curr_frag_ids.push_back(frag_id);
          }
        }
      }
    }
    rowid_lookup_key_ = std::max(rowid_lookup_key_, skip_frag.second);
  }
}

namespace {

bool is_sample_query(const RelAlgExecutionUnit& ra_exe_unit) {
  const bool result = ra_exe_unit.input_descs.size() == 1 &&
                      ra_exe_unit.simple_quals.empty() && ra_exe_unit.quals.empty() &&
                      ra_exe_unit.sort_info.order_entries.empty() &&
                      ra_exe_unit.scan_limit;
  if (result) {
    CHECK_EQ(size_t(1), ra_exe_unit.groupby_exprs.size());
    CHECK(!ra_exe_unit.groupby_exprs.front());
  }
  return result;
}

}  // namespace

bool QueryFragmentDescriptor::terminateDispatchMaybe(
    const RelAlgExecutionUnit& ra_exe_unit,
    const size_t kernel_id) const {
  const auto sample_query_limit =
      ra_exe_unit.sort_info.limit + ra_exe_unit.sort_info.offset;
  if (is_sample_query(ra_exe_unit) && sample_query_limit > 0 &&
      getOuterFragmentTupleSize(kernel_id) >= sample_query_limit) {
    return true;
  }
  return false;
}

void QueryFragmentDescriptor::checkDeviceMemoryUsage(
    const Fragmenter_Namespace::FragmentInfo& fragment,
    const int device_id,
    const size_t num_bytes_for_row) {
  if (g_cluster) {
    // Disabled in distributed mode for now
    return;
  }
  CHECK_GE(device_id, 0);
  tuple_count_per_device_[device_id] += fragment.getNumTuples();
  const size_t gpu_bytes_limit =
      available_gpu_mem_bytes_[device_id] * gpu_input_mem_limit_percent_;
  if (tuple_count_per_device_[device_id] * num_bytes_for_row > gpu_bytes_limit) {
    LOG(WARNING) << "Not enough memory on device " << device_id
                 << " for input chunks totaling "
                 << tuple_count_per_device_[device_id] * num_bytes_for_row
                 << " bytes (available device memory: " << gpu_bytes_limit << " bytes)";
    throw QueryMustRunOnCpu();
  }
}
